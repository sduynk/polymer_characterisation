import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch import Tensor
import numpy as np

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, omega=10):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.omega = omega

    def forward(self, x):
        x = torch.sin(self.fc1(x)*self.omega)
        x = self.fc2(x)
        return x

class FILMLayer(nn.Module):
    def __init__(self, hidden_size, num_features, omega=10):
        super(FILMLayer, self).__init__()

        self.gamma = MLP(1, hidden_size, num_features, omega)
        self.beta = MLP(1, hidden_size, num_features, omega)

    def forward(self, x, condition):
        gamma = self.gamma(condition)[:,:,None, None]
        beta = self.beta(condition)[:,:,None, None]

        return x + (gamma * x + beta)




class Small(nn.Module):
    def __init__(
            self,
            film_hidden_size,
            film_omega,
            width_scale,
            head_hidden_size,
            out_size=1,
            dropout_rate=0.5
            ):
        super().__init__()

        width_1 = int(8 * width_scale)
        width_2 = int(16 * width_scale)
        width_3 = int(32 * width_scale)

        self.out_size = out_size
        self.dropout_rate = dropout_rate

        self.conv1 = nn.Conv2d(3, width_1, 7, 4, padding=3) ## 28
        self.maxpool1 = nn.MaxPool2d(2, stride=2) ## 14
        self.film1 = FILMLayer(film_hidden_size, width_1, film_omega)

        self.conv2 = nn.Conv2d(width_1, width_2, 3, padding=1)
        self.maxpool2 = nn.MaxPool2d(2, stride=2) ## 7
        self.film2 = FILMLayer(film_hidden_size, width_2, film_omega)

        self.conv3 = nn.Conv2d(width_2, width_3, 3, padding=1)
        self.maxpool3 = nn.MaxPool2d(2, stride=2)
        self.film3 = FILMLayer(film_hidden_size, width_3, film_omega)

        self.dropout_1 = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(width_3*7*7, head_hidden_size)
        self.dropout_2 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(head_hidden_size, self.out_size)
    
    def forward(self, x, concentration):
        B = x.shape[0]
        x = self.maxpool1(F.relu(self.conv1(x)))
        x = self.film1(x, concentration)
        x = self.maxpool2(F.relu(self.conv2(x)))
        x = self.film2(x, concentration)
        x = self.maxpool3(F.relu(self.conv3(x)))
        x = self.film3(x, concentration)
        x = x.view(B, -1)
        x = self.dropout_1(x)
        x = self.dropout_2(F.relu(self.fc1(x)))
        x = self.fc2(x)
        x = x.view(x.shape[0], self.out_size)
        return x

    
def init_model(config):
    model = Small(config['film_hidden_size'], config['omega'], config['width'], config['head_hidden_size'], out_size=1, dropout_rate=config['dropout'])
    return model


