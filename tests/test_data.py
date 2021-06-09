from make_dataset import mnist
from torchvision import datasets, transforms
import tensorflow as tf
import torch

def test_data_size():
    train_set, test_set = mnist()
    assert len(train_set) == 60000
    assert len(test_set) == 10000
    assert next(iter(train_set))[0].shape == torch.Size([1,28,28])

    assert (len(torch.unique(train_set.targets))) == 10

test_data_size()
