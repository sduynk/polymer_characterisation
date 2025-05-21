import torch
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error, mean_absolute_error, root_mean_squared_error
from dataloaders import cross_val_particle_size
from utils import set_seed, filter_params
import torch.nn as nn
from ps_models import init_model
import pandas as pd

def train_epoch(train_loader, model, optimizer, scheduler, criterion, device):
    """
    Trains the model for one epoch.
    Args:
        train_loader (DataLoader): DataLoader for the training data.
        model (nn.Module): The model to be trained.
        optimizer (Optimizer): The optimizer used for training.
        scheduler (Scheduler): The learning rate scheduler.
        criterion (Loss): The loss function.
        device (torch.device): The device to run the training on (CPU or GPU).
    Returns:
        float: The average loss for the epoch.
    """
    model.train()
    running_loss = 0.0

    for images, concentrations, particle_sizes in train_loader:
        images = images.to(device)
        concentrations = concentrations.float().to(device).unsqueeze(1)
        particle_sizes = particle_sizes.float().to(device)

        optimizer.zero_grad()

        outputs = model(images, concentrations)
        outputs = outputs.view(-1)

        loss = criterion(outputs, particle_sizes)

        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    return epoch_loss

def predict(model, dataloader, device):
    """
    Perform prediction using the given model and dataloader.
    Args:
        model (torch.nn.Module): The model to use for prediction.
        dataloader (torch.utils.data.DataLoader): DataLoader providing the data.
        device (torch.device): The device to run the model on.
    Returns:
        tuple: A tuple containing:
            - all_outputs (np.ndarray): The raw model outputs.
            - all_denormed_outputs (np.ndarray): The denormalized model outputs.
            - all_targets (np.ndarray): The target particle sizes.
            - all_denormed_targets (np.ndarray): The denormalized target particle sizes.
            - all_concentrations (np.ndarray): The concentrations of the samples.
    """
    model.eval()
    all_outputs = []
    all_denormed_outputs = []
    all_targets = []
    all_denormed_targets = []
    all_concentrations = []

    normalizer = dataloader.dataset.target_transform

    for images, concentrations, particle_sizes in dataloader:
        images = images.to(device)
        concentrations = concentrations.float().to(device).unsqueeze(1)
        particle_sizes = particle_sizes.float().to(device)

        outputs = model(images, concentrations)
        outputs = outputs.view(-1)

        denormed_outputs = normalizer.denormalize(outputs)
        denormed_targets = normalizer.denormalize(particle_sizes)

        all_outputs.append(outputs.detach().cpu().numpy())
        all_denormed_outputs.append(denormed_outputs.detach().cpu().numpy())
        all_targets.append(particle_sizes.detach().cpu().numpy())
        all_denormed_targets.append(denormed_targets.detach().cpu().numpy())
        all_concentrations.append(concentrations.view(-1).detach().cpu().numpy())

    all_outputs = np.concatenate(all_outputs)
    all_denormed_outputs = np.concatenate(all_denormed_outputs)
    all_targets = np.concatenate(all_targets)
    all_denormed_targets = np.concatenate(all_denormed_targets)
    all_concentrations = np.concatenate(all_concentrations)

    return all_outputs, all_denormed_outputs, all_targets, all_denormed_targets, all_concentrations

def test_model(dataloader, model, criterion, device):
    """
    Evaluates the performance of a regression model on a given dataset.
    Args:
        dataloader (DataLoader): DataLoader providing the test data.
        model (nn.Module): The regression model to be evaluated.
        criterion (nn.Module): Loss function used to compute the loss.
        device (torch.device): Device on which to perform computations.
    Returns:
        dict: A dictionary containing various evaluation metrics including loss, denormed loss, R2 score, 
              absolute percentage error, root mean squared error, mean absolute error, and mean squared error.
    """
    all_outputs, all_denormed_outputs, all_targets, all_denormed_targets, all_concentrations = predict(model, dataloader, device)

    loss = criterion(torch.tensor(all_outputs), torch.tensor(all_targets)).item()
    denormed_loss = criterion(torch.tensor(all_denormed_outputs), torch.tensor(all_denormed_targets)).item()

    r2 = r2_score(all_denormed_targets, all_denormed_outputs)
    absolute_percentage_error = mean_absolute_percentage_error(all_denormed_targets, all_denormed_outputs)
    root_mse = root_mean_squared_error(all_denormed_targets, all_denormed_outputs)
    absolute_error = mean_absolute_error(all_denormed_targets, all_denormed_outputs)
    squared_error = mean_squared_error(all_denormed_targets, all_denormed_outputs)

    metrics = {
        'loss': loss,
        'denormed_loss': denormed_loss,
        'r2': r2,
        'absolute_percentage_error': absolute_percentage_error,
        'root_mse': root_mse,
        'absolute_error': absolute_error,
        'squared_error': squared_error
    }

    return metrics

def train_model(train_loader, val_loader, model, optimizer, scheduler, criterion, device, num_epochs=10):
    """
    Trains a model for a specified number of epochs and evaluates it on a validation set.
    Args:
        train_loader (DataLoader): DataLoader for the training data.
        val_loader (DataLoader): DataLoader for the validation data.
        model (nn.Module): The model to be trained.
        optimizer (Optimizer): Optimizer for updating model parameters.
        scheduler (Scheduler): Learning rate scheduler.
        criterion (Loss): Loss function.
        device (torch.device): Device to run the model on (CPU or GPU).
        num_epochs (int, optional): Number of epochs to train the model. Default is 10.
    Returns:
        None
    """
    for epoch in range(num_epochs):
        train_loss = train_epoch(train_loader, model, optimizer, scheduler, criterion, device)
        val_metrics = test_model(val_loader, model, criterion, device)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_metrics['loss']:.4f}, Val R2: {val_metrics['r2']:.4f}, Val MAPE: {val_metrics['absolute_percentage_error']:.4f}, Val RMSE: {val_metrics['root_mse']:.4f}, Val MAE: {val_metrics['absolute_error']:.4f}, Val MSE: {val_metrics['squared_error']:.4f}")

    print("Training complete!")

def train_cross_val(config, with_validation_set=False, save_path=None):
    """
    Trains a model using cross-validation and returns the test results.
    Args:
        config (dict): Configuration dictionary containing model parameters, 
                       optimizer settings, and training options.
        with_validation_set (bool, optional): If True, includes a validation set 
                                              in the cross-validation process which
                                              can be used for hparam search, checkpointing, etc. 
                                              Defaults to False.
    Returns:
        list: A list of dictionaries containing test metrics for each fold, 
              including loss, R2, MAPE, RMSE, MAE, and MSE.
    """
    
    test_results = []

    for fold, (train_loader, val_loader, test_loader, target_normalizer) in enumerate(cross_val_particle_size(config, with_validation_set)):

        set_seed(config['seed']) # seed for reproducibility

        model = init_model(config).to(config['device'])
        params = filter_params(model) # remove weight decay from bias and batchnorm layers
        optimizer = torch.optim.AdamW([
                {'params': params['decay'], 'weight_decay': config['weight_decay']},
                {'params': params['no_decay'], 'weight_decay': 0.0}],
                lr=config['lr'])
        
        criterion = nn.MSELoss() if config['mse'] else nn.L1Loss()
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config['lr'], steps_per_epoch=len(train_loader), epochs=config['num_epochs'], pct_start=0.1)
        
        device = config['device']
        num_epochs = config['num_epochs']

        train_model(train_loader, val_loader, model, optimizer, scheduler, criterion, device, num_epochs)
        test_metrics = test_model(test_loader, model, criterion, device)
        test_results.append(test_metrics)

        if save_path:
            print(f"Saving model for fold {fold}...")
            torch.save({
                'model_state_dict': model.state_dict(),
                'target_normalizer': target_normalizer
            }, f"{save_path}/fold_{fold}.pth")

        print(f"Test (Normalized) Loss: {test_metrics['loss']:.4f}, Test R2: {test_metrics['r2']:.4f}, Test MAPE: {test_metrics['absolute_percentage_error']:.4f}, Test RMSE: {test_metrics['root_mse']:.4f}, Test MAE: {test_metrics['absolute_error']:.4f}, Test MSE: {test_metrics['squared_error']:.4f}")

    return test_results

def full_dataset_predictions(config, with_validation_set=False, save_path=None):
    """
    Generate predictions for the full dataset using cross-validation. Used for generating plots in "figures.ipynb"
    Args:
        config (dict): Configuration dictionary containing model parameters and settings.
        with_validation_set (bool, optional): Whether to include a validation set in the cross-validation. Defaults to False.
    Returns:
        pd.DataFrame: DataFrame containing the outputs, denormed outputs, targets, denormed targets, and concentrations.
    """

    all_outputs = []
    all_denormed_outputs = []
    all_targets = []
    all_denormed_targets = []
    all_concentrations = []

    for fold, (train_loader, val_loader, test_loader, target_normalizer) in enumerate(cross_val_particle_size(config, with_validation_set)):

        set_seed(config['seed'])

        model = init_model(config).to(config['device'])
        params = filter_params(model)
        optimizer = torch.optim.AdamW([
                {'params': params['decay'], 'weight_decay': config['weight_decay']},
                {'params': params['no_decay'], 'weight_decay': 0.0}],
                lr=config['lr'])
        
        criterion = nn.MSELoss() if config['mse'] else nn.L1Loss()
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config['lr'], steps_per_epoch=len(train_loader), epochs=config['num_epochs'], pct_start=0.1)
        
        device = config['device']
        num_epochs = config['num_epochs']

        train_model(train_loader, val_loader, model, optimizer, scheduler, criterion, device, num_epochs)
        
        outputs, denormed_outputs, targets, denormed_targets, concentrations = predict(model, test_loader, device)

        if save_path:
            print(f"Saving model for fold {fold}...")
            torch.save({
                'model_state_dict': model.state_dict(),
                'target_normalizer': target_normalizer
            }, f"{save_path}/fold_{fold}.pth")
        
        all_outputs.append(outputs)
        all_denormed_outputs.append(denormed_outputs)
        all_targets.append(targets)
        all_denormed_targets.append(denormed_targets)
        all_concentrations.append(concentrations)
    
    all_outputs = np.concatenate(all_outputs)
    all_denormed_outputs = np.concatenate(all_denormed_outputs)
    all_targets = np.concatenate(all_targets)
    all_denormed_targets = np.concatenate(all_denormed_targets)
    all_concentrations = np.concatenate(all_concentrations)

    data = {
        'outputs': all_outputs,
        'denormed_outputs': all_denormed_outputs,
        'targets': all_targets,
        'denormed_targets': all_denormed_targets,
        'concentrations': all_concentrations
    }

    df = pd.DataFrame(data)
    return df

