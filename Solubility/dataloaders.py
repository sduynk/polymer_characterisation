import os
import torch
from PIL import Image
from torchvision import transforms
from sklearn.model_selection import GroupKFold
import torch.utils
import numpy as np
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class Log1pTransform:
    def __call__(self, img):
        return torch.log1p(img)

def idx_to_class(idx, num_classes=4):
    if num_classes == 2:
        return ["soluble", "soluble", "insoluble", "insoluble"][idx]
    elif num_classes == 3:
        return ["colloidal", "soluble", "insoluble", "insoluble"][idx]
    elif num_classes == 4:
         return ["colloidal", "soluble", "insoluble", "partialsoluble"][idx]

class SolubilityDataset(Dataset):
    def __init__(self, root, image_paths, targets, transform=None):
        self.root = root
        self.image_paths = [os.path.join(root, image_path) for image_path in image_paths]
        self.class_to_idx = {"colloidal": 0, "soluble": 1, "insoluble": 2, "partialsoluble": 3}
        self.idx_to_class = ["colloidal", "soluble", "insoluble", "partialsoluble"]
        self.targets = [self.class_to_idx[target] for target in targets]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        class_label = self.targets[index]

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, int(class_label)



def get_transforms(config):
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.CenterCrop(config['center_crop']),
        transforms.RandomAffine(degrees=config['degrees'], translate=config['translate'], scale=(config['scale_lower'], config['scale_upper'])),
        transforms.Resize(config['resize']),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), # even with pretrained imagenet weights, 0-1 normalization is better

    ])
    
    test_transform = transforms.Compose([
        transforms.CenterCrop(config['center_crop']),
        transforms.Resize(config['resize']),
        transforms.ToTensor(),
        # transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), # even with pretrained imagenet weights, 0-1 normalization is better
    ])

    return train_transform, test_transform


def prep_data(train_df, val_df, test_df, config):
    train_transform, test_transform = get_transforms(config)

    train_dataset = SolubilityDataset(config['data_dir'], train_df['file'].tolist(), train_df['class'].tolist(), transform=train_transform)
    val_dataset = SolubilityDataset(config['data_dir'], val_df['file'].tolist(), val_df['class'].tolist(), transform=test_transform)
    test_dataset = SolubilityDataset(config['data_dir'], test_df['file'].tolist(), test_df['class'].tolist(), transform=test_transform)

    # Weighted sampler to address class imbalance
    class_counts = np.bincount(train_dataset.targets)
    class_weights = 1./class_counts
    sample_weights = class_weights[train_dataset.targets]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], sampler=sampler, num_workers=4, pin_memory=True, drop_last=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4, pin_memory=True, drop_last=False, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4, pin_memory=True, drop_last=False, persistent_workers=True)

    return train_loader, val_loader, test_loader

def cross_val_solubility(config):
    df = pd.read_csv(os.path.join(config['data_dir'], "annotations.csv"))
    df = df.dropna()

    # unique identifier for each polymer-solvent pair, used for stratified group kfold
    # multiple images exist for each polymer-solvent pair, so we need to ensure that they are not split across train, val, and test
    # This allows us to evaluate the model on unseen polymer-solvent pairs
    df["polymersolvent"] = df["polymer"] + df["solvent"] 

    test_sgkf = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=43) # keeping random states fixed for reproducibility and comparison between models
    val_sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=43) # keeping random states fixed for reproducibility and comparison between models

    for trainval_idx, test_idx in test_sgkf.split(df, df["class"], groups=df["polymersolvent"]):
        trainval_df = df.iloc[trainval_idx]
        test_df = df.iloc[test_idx]

        for train_idx, val_idx in val_sgkf.split(trainval_df, trainval_df['class'], groups=trainval_df['polymersolvent']):
            train_df = trainval_df.iloc[train_idx]
            val_df = trainval_df.iloc[val_idx]
            break

        yield prep_data(train_df, val_df, test_df, config)



