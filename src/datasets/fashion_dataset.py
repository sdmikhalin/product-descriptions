import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class FashionDataset(Dataset):
    def __init__(self, image_dir, df, processor):
        self.image_dir = image_dir
        self.df = df
        self.processor = processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(
            os.path.join(self.image_dir, f"{row['id']}.jpg")
        ).convert('RGB')
        encoding = self.processor(
            images=image,
            text=row['description'],
            padding="max_length",
            max_length=32,
            return_tensors="pt"
        )
        encoding = {k: v.squeeze() for k, v in encoding.items()}
        encoding['labels'] = encoding['input_ids']
        return encoding

