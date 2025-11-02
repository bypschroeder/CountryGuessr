import os
import pycountry
from torch.utils.data import Dataset
from PIL import Image


# Maps image to ID (class) used for training which then later at the prediction has to be converted to country
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
                if fname.lower().endswith('.md'):
                    continue
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
