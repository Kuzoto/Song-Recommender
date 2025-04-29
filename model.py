import torch
import torch.nn as nn
#Future idea expand model to train by predicting genre and then train
#a second model to predict valence based on latent feature vector and
#feed it the predicted latent feature vector of a song from first model
#to predict the valence value of the song. (This will use data from spotify)

def conv(ni, nf, ks=3, stride=1, bias=False):
    return nn.Conv2d(ni, nf, kernel_size=ks, stride=stride, padding=ks//2, bias=bias)

# Convolutional layer with 3x3 convolution, batch normalization, and ReLU activation
def conv_layer(ni, nf, ks=3, stride=1, act=True):
    bn = nn.BatchNorm2d(nf) #stabilize training and prevent overfitting
    layers = [conv(ni, nf, ks, stride=stride), bn]
    act_fn = nn.ReLU(inplace=True)
    if act: layers.append(act_fn)
    return nn.Sequential(*layers)

# ResNet Block with two convolutional layers and a skip connection
class ResBlock(nn.Module):
    def __init__(self, nf):
        super().__init__()
        self.conv1 = conv_layer(nf, nf)
        self.conv2 = conv_layer(nf, nf)
    def forward(self, x):
        return x + self.conv2(self.conv1(x))

# Use average pooling to focus on the entire feature map (i.e. the entire song)
def layer_averpl(ni, nf):
    aver_pl = nn.AvgPool2d(kernel_size=2, stride=2)
    return nn.Sequential(conv_layer(ni, nf), aver_pl)

model = nn.Sequential(
    # convolutional layers (computes the latent feature vector of the spectrogram)
    layer_averpl(1, 64), #64 channels, 64x64 filtered pixels
    ResBlock(64), # ensure information flows to next layer to address vanishing gradients
    layer_averpl(64, 64), #64 channels, 32x32 filtered pixels
    ResBlock(64),
    layer_averpl(64, 128), #128 channels, 16x16 filtered pixels
    ResBlock(128),
    layer_averpl(128, 256), #256 channels, 8x8 filtered pixels
    ResBlock(256),
    layer_averpl(256, 512), #512 channels, 4x4 filtered pixels
    ResBlock(512),
    # dense layers (converts the feature vector into a classification/prediction)
    nn.AdaptiveAvgPool2d((2,2)), # Ensure the song features have the correct size regardless of song length/image size
    nn.Flatten(),
    nn.Linear(2048, 40), # Produce representative latent feature vector
    nn.Linear(40, 8) # Produce genre classification/prediction
)