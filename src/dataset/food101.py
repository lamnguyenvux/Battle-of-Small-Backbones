from torch.utils.data import Dataset
import os
from PIL import Image


class Food101Dataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.image_dir = os.path.join(root_dir, 'images')
        self.label_file = os.path.join(root_dir, 'meta', f'{split}.txt')
        self.data = []

        # Read the data and labels from the txt file
        with open(self.label_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                self.data.append(line.strip())

        self.classes = sorted({d.split('/')[0] for d in self.data})
        self.class_to_idx = {
            cls_name: i for i, cls_name in enumerate(self.classes)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.data[idx] + '.jpg')
        image = Image.open(img_name).convert('RGB')
        label = self.data[idx].split('/')[0]
        label = self.class_to_idx[label]

        if self.transform:
            image = self.transform(image)

        return image, label
