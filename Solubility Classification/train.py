import torch
import copy
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from utils import *
from models import init_model
import torch.nn as nn
from dataloaders import cross_val_solubility
import pandas as pd

def idx_to_class(idx, num_classes=4):
    if num_classes == 2:
        return ["soluble", "soluble", "insoluble", "insoluble"][idx]
    elif num_classes == 3:
        return ["colloidal", "soluble", "insoluble", "insoluble"][idx]
    elif num_classes == 4:
         return ["colloidal", "soluble", "insoluble", "partialsoluble"][idx]


def metrics(gt, preds, num_classes):

    gt = [idx_to_class(g, num_classes) for g in gt]
    preds = [idx_to_class(p, num_classes) for p in preds]

    accuracy = accuracy_score(gt, preds)
    precision = precision_score(gt, preds, average='macro', )
    recall = recall_score(gt, preds, average='macro')
    f1 = f1_score(gt, preds, average='macro')

    accuracy = round(accuracy, 4)
    precision = round(precision, 4)
    recall = round(recall, 4)
    f1 = round(f1, 4)

    return {f"accuracy@{num_classes}": accuracy, f"precision@{num_classes}": precision, f"recall@{num_classes}": recall, f"F1@{num_classes}": f1}


def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    running_loss = 0.0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = torch.nn.functional.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        scheduler.step()

    epoch_loss = running_loss / len(dataloader)
    return epoch_loss


def test_model(dataloader, model, device):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())


    metric_dictionary = metrics(all_labels, all_preds, 4) | metrics(all_labels, all_preds, 3) | metrics(all_labels, all_preds, 2)    

    return metric_dictionary

def train_model(train_dataloader, test_dataloader, model, optimizer, scheduler, device, num_epochs=10):
    best_model_wts = model.state_dict()
    best_f1 = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Train phase
        train_loss = train_epoch(model, train_dataloader, optimizer, scheduler, device)
        print(f'Train Loss: {train_loss:.4f}')

        # Test phase
        metrics_dict = test_model(test_dataloader, model, device)
        print(f'Test Metrics: {metrics_dict}')

        # Deep copy the model
        if metrics_dict['F1@4'] > best_f1:
            best_f1 = metrics_dict['F1@4']
            best_model_wts = copy.deepcopy(model.state_dict())

    print(f'Best F1 Score: {best_f1:.4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model
    
def train_cross_val(config, save_path=None):
    
    test_results = []
    val_results = []

    for fold, (train_loader, val_loader, test_loader) in enumerate(cross_val_solubility(config)):

        set_seed(config['seed']) # seed for reproducibility

        model = init_model(config).to(config['device'])
        params = filter_params(model) # remove weight decay from bias and batchnorm layers
        optimizer = torch.optim.AdamW([
                {'params': params['decay'], 'weight_decay': config['weight_decay']},
                {'params': params['no_decay'], 'weight_decay': 0.0}],
                lr=config['lr'])
        
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config['lr'], steps_per_epoch=len(train_loader), epochs=config['num_epochs'], pct_start=0.1)
        
        device = config['device']
        num_epochs = config['num_epochs']

        train_model(train_loader, val_loader, model, optimizer, scheduler, device, num_epochs)
        val_metrics = test_model(val_loader, model, device)
        test_metrics = test_model(test_loader, model, device)

        val_results.append(val_metrics)
        test_results.append(test_metrics)

        if save_path:
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            print(f"Saving model for fold {fold}...")
            torch.save({
                'model_state_dict': model.state_dict(),
            }, f"{save_path}/fold_{fold}.pth")

        print(f'Test Metrics: {test_metrics}')

    val_results = pd.DataFrame(val_results)
    test_results = pd.DataFrame(test_results)

    return val_results, test_results

def train_no_cv(config, save_path=None):

    # Some hacky code just to train a single model without cross-validation
    # Initially cv was to be used for everything, but this was very costly for hparam optimization
    # Ultimately a decision was made that it would be better to explore more hparams over a single train, val split
    # than to explore fewer params over multiple splits given the time constraints
    
    set_seed(config['seed']) # seed for reproducibility

    for fold, (train_loader, val_loader, test_loader) in enumerate(cross_val_solubility(config)):

        model = init_model(config).to(config['device'])
        params = filter_params(model) # remove weight decay from bias and batchnorm layers
        optimizer = torch.optim.AdamW([
                {'params': params['decay'], 'weight_decay': config['weight_decay']},
                {'params': params['no_decay'], 'weight_decay': 0.0}],
                lr=config['lr'])
        
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config['lr'], steps_per_epoch=len(train_loader), epochs=config['num_epochs'], pct_start=0.1)
        
        device = config['device']
        num_epochs = config['num_epochs']

        train_model(train_loader, val_loader, model, optimizer, scheduler, device, num_epochs)
        val_metrics = test_model(val_loader, model, device)
        test_metrics = test_model(test_loader, model, device)

        if save_path:
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            print("Saving model...")
            torch.save({
                'model_state_dict': model.state_dict(),
            }, f"{save_path}.pth")

        print(f'Test Metrics: {test_metrics}')

        break

    return val_metrics, test_metrics


