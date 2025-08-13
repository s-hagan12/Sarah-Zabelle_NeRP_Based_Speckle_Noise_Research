import numpy as np

import torch
import torch.nn as nn
from math import sqrt

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device("cpu")


############ Input Positional Encoding ############
class Positional_Encoder():
    def __init__(self, params):
        if params['embedding'] == 'gauss':
            self.B = torch.randn((params['embedding_size'], params['coordinates_size'])) * params['scale']
            self.B = self.B.to(device)
        elif params['embedding'] == 'preset':
            self.B = torch.load('saved_gauss_b2.pt')
            self.B = self.B.to(device)
        else:
            raise NotImplementedError

    def embedding(self, x):
        self.B = self.B.to(device)
        x_embedding = (2. * np.pi * x) @ self.B.t()
        x_embedding = torch.cat([torch.sin(x_embedding), torch.cos(x_embedding)], dim=-1)
        return x_embedding



############ Fourier Feature Network ############
class Swish(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)

class FFN(nn.Module):
    def __init__(self, params):
        super(FFN, self).__init__()

        num_layers = params['network_depth']
        hidden_dim = params['network_width']
        input_dim = params['network_input_size']
        output_dim = params['network_output_size']

        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for i in range(1, num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim, output_dim))
        layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        out = self.model(x)
        return out



############ SIREN Network ############
class SirenLayer(nn.Module):
    def __init__(self, in_f, out_f, w0=30, is_first=False, is_last=False):
        super().__init__()
        self.in_f = in_f
        self.w0 = w0
        self.linear = nn.Linear(in_f, out_f)
        self.is_first = is_first
        self.is_last = is_last
        self.init_weights()

    def init_weights(self):
        b = 1 / \
            self.in_f if self.is_first else np.sqrt(6 / self.in_f) / self.w0
        with torch.no_grad():
            self.linear.weight.uniform_(-b, b)

    def forward(self, x):
        x = self.linear(x)
        return x if self.is_last else torch.sin(self.w0 * x)


class SIREN(nn.Module):
    def __init__(self, params):
        super(SIREN, self).__init__()

        num_layers = params['network_depth']
        hidden_dim = params['network_width']
        input_dim = params['network_input_size']
        output_dim = params['network_output_size']

        layers = [SirenLayer(input_dim, hidden_dim, is_first=True)]
        for i in range(1, num_layers - 1):
            layers.append(SirenLayer(hidden_dim, hidden_dim))
        layers.append(SirenLayer(hidden_dim, output_dim, is_last=True))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        out = self.model(x)

        return out

############ SIREN with Batch Normalisation ##############
#From the paper 
#Batch Normalization Alleviates the Spectral Bias in Coordinate Networks (CVPR 2024) & Towards the Spectral bias Alleviation by Normalizations in Coordinate Networks
#https://github.com/Aiolus-X/Norm-INR/blob/main/image%20fitting/bn-29.74/siren.py
#Add MIT license
class SineBN(nn.Module):
    """Sine activation with scaling.

    Args:
        w0 (float): Omega_0 parameter from SIREN paper.
    """
    def __init__(self, w0=1.):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)

