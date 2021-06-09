
from model import MyAwesomeModel
import numpy as np
import torch
from make_dataset import mnist
from torch import nn, optim
import matplotlib.pyplot as plt

def test_Model():
 
    model = MyAwesomeModel()
    train_set, _ = mnist()

    trainloader = torch.utils.data.DataLoader(
                train_set, batch_size=64, shuffle=True)
    for images, labels in trainloader:
            if images.ndim != 4:
                raise ValueError('Expected input to a 4D tensor')
            output = model(images)
            assert images[0].shape == torch.Size([1,28,28])
            assert output[0].shape == torch.Size([10])
            break

