import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class Sine(nn.Module):
    def __init__(self):
        super(Sine, self).__init__()

    def forward(self, x):
        return torch.sin(x)

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, omega=10.0, activation='sine'):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.omega = omega

        if activation == 'sine':
            self.activation = Sine()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'identity':
            self.activation = nn.Identity()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        print(f"MLP initialized with omega: {self.omega}")
        print(f"Using {activation} activation function in the first layer.")


    def forward(self, x):
        x = self.activation(self.fc1(x)*self.omega)
        x = self.fc2(x)
        return x

class FILMLayer(nn.Module):
    def __init__(self, hidden_size, num_features, omega=10, activation='sine'):
        super(FILMLayer, self).__init__()

        self.gamma = MLP(1, hidden_size, num_features, omega, activation)
        self.beta = MLP(1, hidden_size, num_features, omega, activation)

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
            dropout_rate=0.5,
            film_activation='sine',
            ):
        super().__init__()

        width_1 = int(8 * width_scale)
        width_2 = int(16 * width_scale)
        width_3 = int(32 * width_scale)

        self.out_size = out_size
        self.dropout_rate = dropout_rate

        self.conv1 = nn.Conv2d(3, width_1, 7, 4, padding=3) ## 28
        self.maxpool1 = nn.MaxPool2d(2, stride=2) ## 14
        self.film1 = FILMLayer(film_hidden_size, width_1, film_omega, film_activation)

        self.conv2 = nn.Conv2d(width_1, width_2, 3, padding=1)
        self.maxpool2 = nn.MaxPool2d(2, stride=2) ## 7
        self.film2 = FILMLayer(film_hidden_size, width_2, film_omega, film_activation)

        self.conv3 = nn.Conv2d(width_2, width_3, 3, padding=1)
        self.maxpool3 = nn.MaxPool2d(2, stride=2)
        self.film3 = FILMLayer(film_hidden_size, width_3, film_omega, film_activation)

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

class EfficientNetFILM(nn.Module):
    def __init__(
        self,
        film_hidden_size,
        film_omega,
        head_hidden_size,
        out_size=1,
        dropout_rate=0.5,
        film_activation='sine',
        pretrained=True,
    ):
        super().__init__()
        # Load EfficientNet-B0 backbone
        self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        self.backbone.classifier = nn.Identity()  # Remove original classifier

        # Interleave FILM layers after each block in EfficientNet features
        self.blocks = nn.ModuleList()
        self.film_layers = nn.ModuleList()
        for block in self.backbone.features:
            self.blocks.append(block)
            # Get output channels for this block
            out_channels = block[-1].out_channels if hasattr(block[-1], 'out_channels') else block.out_channels
            self.film_layers.append(FILMLayer(film_hidden_size, out_channels, film_omega, film_activation))

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        feature_dim = self.backbone.features[-1][0].out_channels

        self.dropout_1 = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(feature_dim, head_hidden_size)
        self.dropout_2 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(head_hidden_size, out_size)
        self.out_size = out_size

    def forward(self, x, concentration):
        for block, film in zip(self.blocks, self.film_layers):
            x = block(x)
            x = film(x, concentration)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout_1(x)
        x = self.dropout_2(F.relu(self.fc1(x)))
        x = self.fc2(x)
        x = x.view(x.shape[0], self.out_size)
        return x

### Init from config

def init_efficientnet_film_model(config):
    model = EfficientNetFILM(
        film_hidden_size=config['film_hidden_size'],
        film_omega=config['omega'],
        head_hidden_size=config['head_hidden_size'],
        out_size=1,
        dropout_rate=config['dropout'],
        film_activation=config['film_activation'],
        pretrained=True,
    )
    return model
    
def init_model(config):
    model = Small(
        config['film_hidden_size'],
        config['omega'], 
        config['width'], 
        config['head_hidden_size'], 
        out_size=1, 
        dropout_rate=config['dropout'],
        film_activation=config['film_activation']
    )
    return model


