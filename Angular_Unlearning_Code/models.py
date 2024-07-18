import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *
# from .camada_binaria import BinaryLayer
# Below methods to claculate input featurs to the FC layer
# and weight initialization for CNN model is based on the below github repo
# Based on :https://github.com/Lab41/cyphercat/blob/master/Utils/models.py

def size_conv(size, kernel, stride=1, padding=0):
    out = int(((size - kernel + 2 * padding) / stride) + 1)
    return out


def size_max_pool(size, kernel, stride=None, padding=0):
    if stride == None:
        stride = kernel
    out = int(((size - kernel + 2 * padding) / stride) + 1)
    return out

# Calculate in_features for FC layer in Shadow Net
def calc_feat_linear_cifar(size):
    feat = size_conv(size, 3, 1, 1)
    feat = size_max_pool(feat, 2, 2)
    feat = size_conv(feat, 3, 1, 1)
    out = size_max_pool(feat, 2, 2)
    return out


# Calculate in_features for FC layer in Shadow Net
def calc_feat_linear_mnist(size):
    feat = size_conv(size, 5, 1)
    feat = size_max_pool(feat, 2, 2)
    feat = size_conv(feat, 5, 1)
    out = size_max_pool(feat, 2, 2)
    return out


# Parameter Initialization
def init_params(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data)
        nn.init.zeros_(m.bias)

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self,x):
        return x.view(x.size(0), -1)

class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, output_padding=0,
                 activation_fn=nn.ReLU, batch_norm=True, transpose=False):
        if padding is None:
            padding = (kernel_size - 1) // 2
        model = []
        if not transpose:
#             model += [ConvStandard(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding
#                                 )]
            model += [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                bias=not batch_norm)]
        else:
            model += [nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                         output_padding=output_padding, bias=not batch_norm)]
        if batch_norm:
            model += [nn.BatchNorm2d(out_channels, affine=True)]
        model += [activation_fn()]
        super(Conv, self).__init__(*model)

class AllCNN(nn.Module):
    def __init__(self, n_channels=3, num_classes=10, dropout=False, filters_percentage=1., batch_norm=True):
        super(AllCNN, self).__init__()
        n_filter1 = int(96 * filters_percentage)
        n_filter2 = int(192 * filters_percentage)
        self.features = nn.Sequential(
            Conv(n_channels, n_filter1, kernel_size=3, batch_norm=batch_norm),
            Conv(n_filter1, n_filter1, kernel_size=3, batch_norm=batch_norm),
            Conv(n_filter1, n_filter2, kernel_size=3, stride=2, padding=1, batch_norm=batch_norm),
            nn.Dropout(inplace=True) if dropout else Identity(),
            Conv(n_filter2, n_filter2, kernel_size=3, stride=1, batch_norm=batch_norm),
            Conv(n_filter2, n_filter2, kernel_size=3, stride=1, batch_norm=batch_norm),
            Conv(n_filter2, n_filter2, kernel_size=3, stride=2, padding=1, batch_norm=batch_norm),  # 14
            nn.Dropout(inplace=True) if dropout else Identity(),
            Conv(n_filter2, n_filter2, kernel_size=3, stride=1, batch_norm=batch_norm),
            Conv(n_filter2, n_filter2, kernel_size=1, stride=1, batch_norm=batch_norm),
            nn.AvgPool2d(8),
            Flatten(),
        )
        self.classifier = nn.Sequential(
                nn.Linear(n_filter2, num_classes),
            )
        # self.

    def forward(self, x):
        features = self.features(x)
        output = self.classifier(features)
        return output

#####################################################
# Define Target, Shadow and Attack Model Architecture
#####################################################

# Target Model
class TargetNet(nn.Module):
    def __init__(self, input_dim, hidden_layers, size, out_classes):
        super(TargetNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=input_dim, out_channels=hidden_layers[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_layers[0]),
            # nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_layers[0], out_channels=hidden_layers[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_layers[1]),
            # nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        features = calc_feat_linear_cifar(size)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear((features ** 2 * hidden_layers[1]), hidden_layers[2]),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_layers[2], out_classes)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.classifier(out)
        return out

# Shadow Model mimicking target model architecture, for our implememtation is different than target
class ShadowNet(nn.Module):
    def __init__(self, input_dim, hidden_layers, size, out_classes):
        super(ShadowNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=input_dim, out_channels=hidden_layers[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_layers[0]),
            # nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_layers[0], out_channels=hidden_layers[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_layers[1]),
            # nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        features = calc_feat_linear_cifar(size)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear((features ** 2 * hidden_layers[1]), hidden_layers[2]),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_layers[2], out_classes)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.classifier(out)
        return out

# Target/Shadow Model for MNIST
class MNISTNet(nn.Module):
    def __init__(self, input_dim, n_hidden, out_classes=10, size=28):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=input_dim, out_channels=n_hidden, kernel_size=5),
            nn.BatchNorm2d(n_hidden),
            # nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=n_hidden, out_channels=n_hidden * 2, kernel_size=5),
            nn.BatchNorm2d(n_hidden * 2),
            # nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        features = calc_feat_linear_mnist(size)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(features ** 2 * (n_hidden * 2), n_hidden * 2),
            nn.ReLU(inplace=True),
            nn.Linear(n_hidden * 2, out_classes)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.classifier(out)
        return out

#Attack MLP Model
class AttackMLP(nn.Module):
    def __init__(self, input_size, hidden_size=64,out_classes=2):
        super(AttackMLP, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, out_classes)
        )
    def forward(self, x):
        out = self.classifier(x)
        return out

class BadNet(nn.Module):

    def __init__(self, input_channels, output_num):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=16, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        fc1_input_features = 800 if input_channels == 3 else 512
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=fc1_input_features, out_features=512),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=512, out_features=output_num),
            nn.Softmax(dim=-1)
        )
        self.dropout = nn.Dropout(p=.5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


"""
    The autoencoder has the layer size [10, 7, 4, 7, 10] for Cifar10.  
"""
class EncoderFC(nn.Module):

    def __init__(self):
        super(EncoderFC, self).__init__()

        self.fc1 = nn.Linear(10, 7)
        self.af1 = nn.Relu()
        self.bn1 = nn.BatchNorm1d(7)
        self.fc2 = nn.Linear(7, 4)
        self.af2 = nn.Relu()
        self.bn2 = nn.BatchNorm1d(4)

    def forward(self, x):

        x = self.fc1(x)
        x = self.af1(x)
        x = self.bn1(x)
        x = self.fc2(x)
        x = self.af2(x)
        x = self.bn2(x)
        return x

class DecoderFC(nn.Module):
    """
    FC Decoder composed by 3 512-units fully-connected layers
    """
    def __init__(self):
        super(DecoderFC, self).__init__()

        self.fc1 = nn.Linear(4, 7)
        self.af1 = nn.Relu()
        self.bn1 = nn.BatchNorm1d(7)
        self.fc2 = nn.Linear(7, 10)
        self.af2 = nn.Relu()
        self.bn2 = nn.BatchNorm1d(10)

    def forward(self, x):
        """
        :param x: encoded features
        :return:
        """
        x = self.fc1(x)
        x = self.af1(x)
        x = self.bn1(x)
        x = self.fc2(x)
        x = self.af2(x)
        x = self.bn2(x)
        return x

class AE(nn.Module):

    def __init__(self):
        super(AE, self).__init__()

        self.fc_encoder = EncoderFC()
        self.fc_decoder = DecoderFC()

    def forward(self, x, num_pass=0):
        r = self.fc_encoder(x)
        out = self.fc_decoder(r)
        return out