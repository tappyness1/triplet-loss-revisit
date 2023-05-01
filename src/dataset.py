import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import Compose, Resize, ToTensor
import matplotlib.pyplot as plt


def get_load_data(root = "data", dataset = "FashionMNIST", download = False):

    if dataset == "FashionMNIST":
        training_data = datasets.FashionMNIST(
            root=root,
            train=True,
            download=False,
            transform=ToTensor()
        )

        test_data = datasets.FashionMNIST(
            root=root,
            train=False,
            download=False,
            transform=ToTensor()
        )
    
    elif dataset == "Flowers102":
        training_data = datasets.Flowers102(
            root=root,
            split="train",
            download=False,
            transform=Compose([Resize((128,128)), ToTensor()]) 
        )

        test_data = datasets.Flowers102(
            root=root,
            split = "test",
            download=False,
            transform=Compose([Resize((128,128)), ToTensor()])
        )
    return training_data, test_data

if __name__ == "__main__":
    train, test = get_load_data()
   
    # img, label = train[1]
    # print (label)

    # for i in train:
    #     print (i[1])

    print (len(test))