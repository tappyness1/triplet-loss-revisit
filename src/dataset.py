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
            download=download,
            transform=ToTensor()
        )

        test_data = datasets.FashionMNIST(
            root=root,
            train=False,
            download=download,
            transform=ToTensor()
        )
    
    elif dataset == "Flowers102":
        training_data = datasets.Flowers102(
            root=root,
            # test has larger dataset hence use it
            split="test",
            download=download,
            transform=Compose([Resize((128,128)), ToTensor()]) 
        )

        test_data = datasets.Flowers102(
            root=root,
            # test has larger dataset so use validation for testing.
            split = "val",
            download=download,
            transform=Compose([Resize((128,128)), ToTensor()])
        )
    return training_data, test_data

if __name__ == "__main__":
    import ssl

    ssl._create_default_https_context = ssl._create_unverified_context

    train, test = get_load_data(dataset = "Flowers102", download = True)
   
    # img, label = train[1]
    # print (label)

    # for i in train:
    #     print (i[1])

    # print (len(test))