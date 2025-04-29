from torchsummary import summary
from model import model
import torch
import torch.nn as nn

train_model = model.to("cuda")
summary(train_model, input_size=(1, 128, 128))

removed = list(model.children())[:-1]
rec_model = nn.Sequential(*removed)

summary(rec_model, input_size=(1, 128, 128))