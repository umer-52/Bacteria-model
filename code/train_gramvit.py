import os
import argparse
import time
from pathlib import Path

import torch
from timm import create_model

# Ensure model registry is loaded
import modeling_finetune  # noqa: F401

from datasets import create_downstream_dataset
from engine_for_finetuning import train_one_epoch, evaluate, get_handler
import utils
import optim_factory


class CpuScaler:
    """Fallback scaler for CPU training to satisfy engine API when AMP is unavailable."""
    state_dict_key = "cpu_scaler"

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        loss.backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None and parameters is not None:
                torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        return None

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        return


def build_args():
    parser = argparse.ArgumentParser("GramViT fine-tuning")

    # Data/task
    parser.add_argument('--task', default='gs_classification', type=str)
    parser.add_argument('--data_path', required=True, type=str)
    parser.add_argument('--image_dir', required=True, type=str)
    parser.add_argument('--k_fold', default=0, type=int)

    # Model
    parser.add_argument('--model', default='longvit_small_patch32_4096_gs_classification', type=str)
    parser.add_argument('--input_size', default=4096, type=int)
    parser.add_argument('--drop_path', default=0.1, type=float)
    parser.add_argument('--finetune', default='', type=str, help='path to checkpoint to finetune from')

    # Train
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--eval_batch_size', default=None, type=int)
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--weight_decay', default=0.05, type=float)
    parser.add_argument('--warmup_epochs', default=5, type=int)
    parser.add_argument('--update_freq', default=1, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_mem', action='store_true', default=True)
    parser.add_argument('--randaug', action='store_true', default=True)
    parser.add_argument('--cached_randaug', action='store_true', default=False)
    parser.add_argument('--seq_parallel', action='store_true', default=False)

    # Optimizer
    parser.add_argument('--opt', default='adamw', type=str)
    parser.add_argument('--opt_eps', default=1e-8, type=float)
    parser.add_argument('--opt_betas', default=(0.9, 0.999), type=tuple)
    parser.add_argument('--clip_grad', default=None, type=float)

    # Logging / Checkpoints
    parser.add_argument('--output_dir', required=True, type=str)
    parser.add_argument('--log_dir', required=True, type=str)
    parser.add_argument('--save_ckpt_freq', default=5, type=int)
    parser.add_argument('--auto_resume', action='store_true', default=True)
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--seed', default=42, type=int)

    # Early stopping
    parser.add_argument('--early_stop_patience', default=20, type=int)
    parser.add_argument('--early_stop_metric', default='wsi_f1', type=str)

    return parser.parse_args()


def main():
    args = build_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    print("🚀 Starting GramViT fine-tuning...")
    print(f"📊 Data root: {args.data_path}")
    print(f"🖼️ Images dir: {args.image_dir}")
    print(f"🧠 Model: {args.model}")

    utils.init_distributed_mode(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)

    # Data
    train_loader, val_loader = create_downstream_dataset(args, is_eval=False)

    # Model
    print("Creating model...")
    model = create_model(args.model, pretrained=False, drop_path_rate=args.drop_path)
    model.to(device)

    model_without_ddp = model

    # Optimizer
    optimizer = optim_factory.create_optimizer(args, model)

    # LR schedule
    num_training_steps_per_epoch = len(train_loader)
    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.lr * 1e-2, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, sched_type='cos'
    )

    # Scaler (CPU fallback)
    if torch.cuda.is_available():
        loss_scaler = utils.NativeScalerWithGradNormCount()
    else:
        loss_scaler = CpuScaler()

    # Logging
    log_writer = utils.TensorboardLogger(args.log_dir)

    # Optionally load finetune weights
    if args.finetune:
        try:
            utils.load_model_and_may_interpolate(
                args.finetune, model, model_key='model|teacher|module', model_prefix='', is_eval=False
            )
        except Exception as e:
            print(f"[WARN] Failed to load finetune weights: {e}")

    # Auto-resume
    utils.auto_load_model(args, model, model_without_ddp, optimizer, loss_scaler, model_ema=None)

    # Handler
    handler = get_handler(args)

    best_metric = -float('inf')
    best_epoch = -1
    patience_counter = 0
    metric_name = args.early_stop_metric

    start_time = time.time()

    for epoch in range(getattr(args, 'start_epoch', 0), args.epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)

        print(f"\n📚 Epoch {epoch+1}/{args.epochs}")
        train_stats = train_one_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            device=device,
            handler=handler,
            epoch=epoch,
            start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values,
            loss_scaler=loss_scaler,
            max_norm=args.clip_grad if args.clip_grad is not None else 0,
            update_freq=args.update_freq,
            model_ema=None,
            log_writer=log_writer,
            task=args.task,
            seq_parallel=args.seq_parallel,
        )

        if (epoch + 1) % args.save_ckpt_freq == 0:
            utils.save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler, model_ema=None)

        test_stats, primary_metric = evaluate(val_loader, model, device, handler)
        chosen_metric = test_stats.get(metric_name, test_stats.get(primary_metric))
        if chosen_metric is None:
            chosen_metric = test_stats.get('acc', None)

        # Log hparams + metrics on final epoch or improvement
        log_writer.update(head='val', **{k: v for k, v in test_stats.items() if isinstance(v, (int, float))})
        log_writer.set_step()

        improved = chosen_metric is not None and chosen_metric > best_metric
        if improved:
            best_metric = chosen_metric
            best_epoch = epoch
            patience_counter = 0
            # Save best
            torch.save({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'args': vars(args),
                'best_metric': best_metric,
            }, os.path.join(args.output_dir, 'checkpoint-best.pth'))
            print(f"✅ New best {metric_name}: {best_metric:.4f} at epoch {epoch}")
        else:
            patience_counter += 1
            print(f"⏳ No improvement in {metric_name}. Patience {patience_counter}/{args.early_stop_patience}")
            if patience_counter >= args.early_stop_patience:
                print("🛑 Early stopping triggered.")
                break

    total_time = time.time() - start_time
    print(f"\n🎉 Training completed. Best {metric_name}: {best_metric:.4f} at epoch {best_epoch}")
    print(f"📝 Artifacts in: {args.output_dir}")


if __name__ == '__main__':
    main()


