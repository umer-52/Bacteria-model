import os

dataset_path = r"f:\GramViT-master\Bacteria_Dataset"
if not os.path.exists(dataset_path):
    print("Dataset path not found")
    exit()

files = os.listdir(dataset_path)
classes = set()

for f in files:
    if f.lower().endswith(('.jpg', '.tif', '.png', '.jpeg')):
        # Format seems to be Genus.species_...
        # We split by '_' and take the first part?
        # Or look for the pattern like 'Acinetobacter.baumanii'
        # The list_dir showed 'Acinetobacter.baumanii_0001_1.jpg'
        # and 'Actinomyces.israeli_0001.jpg'
        parts = f.split('_')
        if len(parts) > 0:
            potential_class = parts[0]
            # Ensure it looks like Genus.species
            if '.' in potential_class:
                classes.add(potential_class)

sorted_classes = sorted(list(classes))
print(f"Found {len(sorted_classes)} classes:")
print(sorted_classes)
