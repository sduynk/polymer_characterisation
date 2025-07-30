from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch
from PIL import Image
import os
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
import numpy as np
from torch.utils.data import WeightedRandomSampler

class Normalizer(object):
    """
    A class used to normalize and denormalize data using the provided mean and standard deviation.
    Attributes:
        mean (float): The mean value used for normalization.
        std (float): The standard deviation value used for normalization.
    Methods:
        state_dict(): Returns a dictionary containing the mean and standard deviation.
        load_state_dict(state_dict): Loads the mean and standard deviation from a dictionary.
        denormalize(target): Denormalizes the target value.
        __call__(target): Normalizes the target value.
        __repr__(): Returns a string representation of the Normalizer object.
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def state_dict(self):
        return {'mean': self.mean, 'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']

    def denormalize(self, target):
        return target * self.std + self.mean

    def __call__(self, target):
        return (target - self.mean) / self.std

    def __repr__(self):
        return self.__class__.__name__ + f"(mean={self.mean}, std={self.std})"
    
class LogNormalizer(object):
    """
    A class used to normalize and denormalize data using log transformation.
    Methods:
        state_dict(): Returns an empty dictionary (no state to save).
        load_state_dict(state_dict): Does nothing (no state to load).
        denormalize(target): Denormalizes the target value using exp and subtracting 1.
        __call__(target): Normalizes the target value using log and adding 1.
        __repr__(): Returns a string representation of the LogNormalizer object.
    """
    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass

    def denormalize(self, target):
        if isinstance(target, torch.Tensor):
            return torch.exp(target) - 1
        else:
            return np.exp(target) - 1

    def __call__(self, target):
        if isinstance(target, torch.Tensor):
            return torch.log(target + 1)
        else:
            return np.log(target + 1)

    def __repr__(self):
        return self.__class__.__name__ + "()"
    

def calculate_weights(df, target, bins='auto'):
    """
    Calculate weights for each sample based on the inverse of the histogram bin counts.
    Args:
        df (pd.DataFrame): DataFrame containing the data.
        target (str): The target variable.
    Returns:
        np.array: Array of weights for each sample.
    """
    hist, bin_edges = np.histogram(df[target], bins=8)
    bin_indices = np.digitize(df[target], bin_edges[:-1])
    weights = 1.0 / hist[bin_indices - 1]
    return weights


class PSDataset(Dataset):
    """
    A custom dataset class for loading images and their corresponding conditions and targets.
    Args:
        root (str): Root directory path where images are stored.
        dataframe (pd.DataFrame): DataFrame containing image paths and associated metadata.
        target (str, optional): Column name in the dataframe for the target variable. Default is "particle_size".
        condition (str, optional): Column name in the dataframe for the condition variable. Default is "concentration".
        transform (callable, optional): A function/transform to apply to the images.
        target_transform (callable, optional): A function/transform to apply to the target variable. For example Normalizer
    Methods:
        __len__(): Returns the total number of samples in the dataset.
        __getitem__(idx): Returns the image, condition, and target for a given index.
    """
    def __init__(self, root, dataframe, target="particle_size", condition="concentration", transform=None, target_transform=None):
        self.root = root
        self.dataframe = dataframe.to_dict('records')
        self.transform = transform
        self.target_transform = target_transform

        self.target = target
        self.condition = condition

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        image_path = self.dataframe[idx]['image_path']
        condition = self.dataframe[idx][self.condition]
        target = self.dataframe[idx][self.target]

        image = Image.open(os.path.join(self.root, image_path)).convert('RGB')

        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            target = self.target_transform(target)

        return image, condition, target
        
def prep_data(config, train_df, val_df, test_df, target="particle_size", condition="concentration"):
    """
    Prepares data loaders for training, validation, and testing datasets.
    Args:
        train_df (pd.DataFrame): DataFrame containing the training data.
        val_df (pd.DataFrame): DataFrame containing the validation data.
        test_df (pd.DataFrame): DataFrame containing the test data.
        target (str, optional): The target variable to predict. Defaults to "particle_size".
        condition (str, optional): The condition variable. Defaults to "concentration".
    Returns:
        tuple: A tuple containing the training, validation, and test data loaders, and the target normalizer.
    """

    assert not any(train_df['unique_id'].isin(val_df['unique_id'])), "Train and validation sets have overlapping entries"
    assert not any(train_df['unique_id'].isin(test_df['unique_id'])), "Train and test sets have overlapping entries"
    # assert not any(val_df['unique_id'].isin(test_df['unique_id'])), "Validation and test sets have overlapping entries" #### Ignoring this check as sometimes val_set == test_set (intentionally) in this code

    target_normalizer = Normalizer(train_df[target].mean(), train_df[target].std())

    train_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    train_dataset = PSDataset(config['data_dir'], train_df, target=target, condition=condition, transform=train_transform, target_transform=target_normalizer)
    val_dataset = PSDataset(config['data_dir'], val_df, target=target, condition=condition, transform=train_transform, target_transform=target_normalizer) 
    test_dataset = PSDataset(config['data_dir'], test_df, target=target, condition=condition, transform=train_transform, target_transform=target_normalizer)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=True, num_workers=4, persistent_workers=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4, persistent_workers=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4, persistent_workers=True)

    return train_loader, val_loader, test_loader, target_normalizer

def cross_val_particle_size(config, with_validation_set=False):
    """
    Perform cross-validation on particle size data.
    This function reads particle size data from a CSV file, creates unique IDs for each data point,
    and performs cross-validation using GroupKFold and GroupShuffleSplit. It yields prepared data
    for training, validation, and testing.
    Args:
        with_validation_set (bool): If True, includes a validation set in the cross-validation process.
                                    If False, only train and test sets are used. (Note that the test 
                                    set is used as the validation set in this case.)
    Yields:
        tuple: Prepared data for training, validation (if with_validation_set is True), and testing.
    """
    df = pd.read_csv(os.path.join(config['data_dir'], 'annotations.csv'))

    # Min-max scale the concentration column
    min_conc = df['concentration'].min()
    max_conc = df['concentration'].max()
    df['concentration'] = (df['concentration'] - min_conc) / (max_conc - min_conc)

    df['unique_id'] = df.apply(lambda row: f"{row['concentration']}_{row['particle_size']}", axis=1)
    
    gkf = GroupKFold(n_splits=5)
    gss = GroupShuffleSplit(n_splits=1, random_state=42, test_size=0.2)

    if with_validation_set:

        for train_idx, test_idx in gkf.split(df, groups=df['unique_id']):
            temp_df = df.iloc[train_idx]
            for train_idx, val_idx in gss.split(temp_df, groups=temp_df['unique_id']):
                train_df = temp_df.iloc[train_idx]
                val_df = temp_df.iloc[val_idx]
            test_df = df.iloc[test_idx]

            yield prep_data(config, train_df, val_df, test_df)
    else:

        for train_idx, test_idx in gkf.split(df, groups=df['unique_id']):
            train_df = df.iloc[train_idx]
            test_df = df.iloc[test_idx]

            yield prep_data(config, train_df, test_df, test_df)
