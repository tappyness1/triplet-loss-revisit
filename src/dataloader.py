from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import pandas as pd
import random
from src.dataset import get_load_data
import torch 
from tqdm import tqdm

def generate_train_set(train):
    train_set = []
    classes = [i[1] for i in train]
    classes = np.array(classes)
    for i in tqdm(range(len(train))):

        # positive_list = [j for j in range(len(classes)) if classes[j] == train[i][1]]
        positive_list = np.argwhere(classes == train[i][1])
        positive_ind = list(random.choice(positive_list))[0]
        while positive_ind == i:
            positive_ind = list(random.choice(positive_list))[0]
        negative_list = np.argwhere(classes != train[i][1])
        negative_ind = list(random.choice(negative_list))[0]
        train_set.append({"anchor": train[i][0], "positive": train[positive_ind][0], "negative": train[negative_ind][0], "anchor_label": train[i][1]})
    
    return train_set

class TripletLossDataset(Dataset):
    def __init__(self, train_set = None, train=True, root = "data"):
        
        self.is_train = train
        
        if self.is_train:  
            if train_set is None:  
                train, _ = get_load_data(root = root)
                self.train_set = generate_train_set(train)
            else:
                self.train_set = train_set
        else:
            _, self.test = get_load_data(root = root)
        
    def __len__(self):
        if self.is_train: 
            return len(self.train_set)
        else: 
            return len(self.test)
    
    def __getitem__(self, item):
        
        if self.is_train:
            data = self.train_set[item]
            
            return data['anchor'], data['positive'], data['negative'], data['anchor_label']
        
        else:
            return self.test[item][0], self.test[item][1]