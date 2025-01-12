import os
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Subset, random_split

from lightning.pytorch import LightningDataModule
from typing import Callable, Literal, Optional

from src.utils import validate_literal_types
from .food101 import Food101Dataset


DATASET_ZOO = Literal[
    'cifar10', 'cifar100', 'cub200', 'dtd', 
    'eurosat', 'flowers102', 'food101', 'plantclef'
]


class DataModule(LightningDataModule):
    def __init__(self, dataset_type: DATASET_ZOO, root: str, batch_size: int = 32):
        super().__init__()
        validate_literal_types(dataset_type, DATASET_ZOO)
        self.dataset_type = dataset_type
        self.root = root
        self.batch_size = batch_size

    def prepare_data(self):
        if self.dataset_type == "cifar10":
            transform_train = transforms.Compose([
                transforms.TrivialAugmentWide(),
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
            ])

            transform_test = transforms.Compose([
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4941, 0.4853, 0.4507), (0.2468, 0.2430, 0.2618))
            ])

            self.trainset = torchvision.datasets.CIFAR10(
                root=self.root, train=True, download=True, transform=transform_train)

            self.testset = torchvision.datasets.CIFAR10(
                root=self.root, train=False, download=True, transform=transform_test)

        elif self.dataset_type == "cifar100":
            transform_train = transforms.Compose([
                transforms.TrivialAugmentWide(),
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
            ])

            transform_test = transforms.Compose([
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4941, 0.4853, 0.4507), (0.2468, 0.2430, 0.2618))
            ])

            self.trainset = torchvision.datasets.CIFAR100(
                root='./data', train=True, download=True, transform=transform_train)

            self.testset = torchvision.datasets.CIFAR100(
                root='./data', train=False, download=True, transform=transform_test)

        elif self.dataset_type == "cub200":
            transform_test = transforms.Compose([
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4867, 0.4990, 0.4287), (0.2273, 0.2221, 0.2613))
            ])

            transform_train = transforms.Compose([
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4855, 0.4993, 0.4318), (0.2274, 0.2227, 0.2613))
            ])

            self.trainset = torchvision.datasets.ImageFolder(
                root=os.path.join(self.root, 'train'), transform=transform_train)

            self.testset = torchvision.datasets.ImageFolder(
                root=os.path.join(self.root, 'test'), transform=transform_test)

        elif self.dataset_type == "dtd":
            transform_train = transforms.Compose([
                transforms.TrivialAugmentWide(),
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])

            transform_test = transforms.Compose([
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
            self.trainset = create_dtd_dataset(
                data_dir=self.root, train=True, transform=transform_train)
            self.testset = create_dtd_dataset(
                data_dir=self.root, train=False, transform=transform_test)

        elif self.dataset_type == "eurosat":
            transform_test = transforms.Compose([
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.3436, 0.3801, 0.4076),
                    std=(0.2014, 0.1354, 0.1134)
                )
            ])

            transform_train = transforms.Compose([
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.3443, 0.3803, 0.4079),
                    std=(0.2007, 0.1347, 0.1133)
                )
            ])

            self.trainset = torchvision.datasets.ImageFolder(
                root=os.path.join(self.root, 'train'), transform=transform_train)

            self.testset = torchvision.datasets.ImageFolder(
                root=os.path.join(self.root, 'test'), transform=transform_test)

        elif self.dataset_type == "flowers102":
            transform_train = transforms.Compose([
                transforms.Resize([224, 224]),
                transforms.TrivialAugmentWide(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4326, 0.3817, 0.2961), (0.2928, 0.2445, 0.2716))
            ])

            transform_test = transforms.Compose([
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4367, 0.3762, 0.2849), (0.2958, 0.2430, 0.2661))
            ])

            self.trainset = torchvision.datasets.Flowers102(
                root=self.root, split='train',  transform=transform_train, download=True)

            self.testset = torchvision.datasets.Flowers102(
                root=self.root, split='val',  transform=transform_test, download=True)

        elif self.dataset_type == "food101":
            transform_train = transforms.Compose([
                transforms.TrivialAugmentWide(),
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5453, 0.4438, 0.3439), (0.2698, 0.2722, 0.2768))
            ])

            transform_test = transforms.Compose([
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5486, 0.4471, 0.3466), (0.2649, 0.2689, 0.2760))
            ])

            self.trainset = Food101Dataset(
                root_dir=self.root,
                split='train',
                transform=transform_train
            )

            self.testset = Food101Dataset(
                root_dir=self.root,
                split='test',
                transform=transform_test
            )

        elif self.dataset_type == "plantclef":
            transform_test = transforms.Compose([
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4221, 0.4473, 0.3362), (0.2345, 0.2329, 0.2433))
            ])

            transform_train = transforms.Compose([
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4298, 0.4588, 0.3523), (0.2322, 0.2313, 0.2459))
            ])

            self.trainset = torchvision.datasets.ImageFolder(
                root=os.path.join(self.root, 'train'), transform=transform_train)

            self.testset = torchvision.datasets.ImageFolder(
                root=os.path.join(self.root, 'test'), transform=transform_test)

        self.valset, self.trainset = random_split(
            dataset=self.trainset,
            lengths=(0.2, 0.8)
        )


    def train_dataloader(self):
        return DataLoader(
            dataset=self.trainset,
            batch_size=self.batch_size,
            num_workers=get_cpu(),
            shuffle=True,
            pin_memory=True,
            drop_last=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            dataset=self.valset,
            batch_size=self.batch_size,
            num_workers=get_cpu(),
            shuffle=False,
            pin_memory=True,
            drop_last=False
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.testset,
            batch_size=self.batch_size,
            num_workers=get_cpu(),
            shuffle=False,
            pin_memory=True,
            drop_last=False
        )


def load_split_indices(data_dir, split='train'):
    split_file = os.path.join(data_dir, 'labels', f'{split}1.txt')
    with open(split_file, 'r') as f:
        lines = f.readlines()
    indices = [
        os.path.join(data_dir, 'images', line.strip())
        for line in lines
    ]
    return indices


def create_dtd_dataset(data_dir: str, train: bool, transform: Optional[Callable] = None):
    dataset = torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'images'), transform=transform)
    split = 'train' if train else 'test'
    split_indices = load_split_indices(data_dir, split)
    # Map the split indices to dataset indices
    dataset_indices = [dataset.samples.index(
        (idx, dataset.class_to_idx[idx.split('/')[-2]])) for idx in split_indices]
    split_dataset = Subset(dataset, dataset_indices)
    return split_dataset


def get_cpu() -> int:
    num_cpu = os.cpu_count()
    if num_cpu is None:
        num_cpu = 0

    return num_cpu
