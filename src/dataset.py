import os
import pycountry
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image


# Maps image to ID used for training which then later at the prediction has to be converted to country
class StreetViewDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.label_map = {}
        self._prepare_samples()

    def _prepare_samples(self):
        codes = sorted(os.listdir(self.root_dir))
        for idx, code in enumerate(codes):
            country_name = pycountry.countries.get(alpha_2=code).name
            self.label_map[country_name] = idx
            code_path = os.path.join(self.root_dir, code)
            for fname in os.listdir(code_path):
                img_path = os.path.join(code_path, fname)
                self.samples.append((img_path, idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


# Example of how to use
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)

dataset = StreetViewDataset("data/raw/streetview_images", transform=transform)
total_size = len(dataset)
train_size = int(0.7 * total_size)
val_size = int(0.15 * total_size)
test_size = total_size - (train_size + val_size)
train_set, val_set, test_set = random_split(
    dataset,
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42),
)

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32)
test_loader = DataLoader(test_set, batch_size=32)

# Example for one sample
image, label = dataset[1823]
print(image)
print(label)

# Example how to convert to country string
idx_to_name = {v: k for k, v in dataset.label_map.items()}
pred_country = idx_to_name[label]
print(pred_country)

# Example of dataloader sample
for batch_images, batch_labels in train_loader:
    print("Batch images shape:", batch_images.shape)
    print("Batch labels:", batch_labels)

    idx_to_name = {v: k for k, v in dataset.label_map.items()}
    for lbl in batch_labels[:5]:
        print("Label:", lbl.item(), "Country:", idx_to_name[lbl.item()])

    break
