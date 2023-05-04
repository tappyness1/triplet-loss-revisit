import torch.nn as nn
import torch.optim as optim
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def validation(model, val_set):
    model.eval()
    val_dataloader = DataLoader(val_set, batch_size=20)
    loss_function = nn.CrossEntropyLoss()
    y_true = []
    y_pred = []

    with tqdm(val_dataloader) as tepoch:
        for imgs, labels in tepoch:
            y_true.extend(labels.numpy())
            with torch.no_grad():
                out = model(imgs)
            y_pred.extend(torch.argmax(out,1).cpu().numpy())
            loss = loss_function(out, labels)    

    cm = pd.DataFrame(confusion_matrix(y_true, y_pred))
    cm.to_csv("val_results/results.csv")
    print (cm)

def get_embeddings(model, val_set):
    model.eval()
    val_dataloader = DataLoader(val_set, batch_size=20)
    embeddings = []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with tqdm(val_dataloader) as tepoch:
        for imgs, _ in tepoch:
            with torch.no_grad():
                out = model(imgs.to(device))
            embeddings.extend(out.cpu().numpy())
    embeddings = np.array(embeddings)
    return embeddings

def get_dist(current_embeddings, all_embeddings, k = 10):
    dist = np.linalg.norm(all_embeddings - current_embeddings, axis = 1)
    idx = np.argsort(dist)[:k]
    return idx

def visualise(model, dataset, embeddings= None, cmap = "gray",  input_idx = 0, k = 10):
    if embeddings is None:
        embeddings = get_embeddings(model, dataset)
    idx = get_dist(embeddings, embeddings[input_idx], k)
    fig=plt.figure(figsize=(6,10))
    columns = 2
    rows = int((k+2)/2)
    fig.add_subplot(rows, columns, 1)
    if cmap == "gray":
        plot_input_img = plt.imshow(dataset[input_idx][0].squeeze(), cmap = cmap)
    else:
        plot_input_img = plt.imshow(dataset[input_idx][0].permute(1,2,0).squeeze(), cmap = cmap)
    for i in range(3, k+3):
        fig.add_subplot(rows, columns, i)
        if cmap == "gray":
            plt.imshow(dataset[idx[i-3]][0].squeeze(), cmap = cmap)  
        else:
            plt.imshow(dataset[idx[i-3]][0].permute(1,2,0).squeeze(), cmap = cmap)
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()

def generate_scatterplot(model, dataset, embeddings = None):

    labels = [dataset[i][1] for i in range(len(dataset))]

    if embeddings is None: 
        model.eval()
        embeddings = get_embeddings(model, dataset)

    labels = np.array(labels)

    plt.figure(figsize=(15, 10), facecolor="azure")
    for label in np.unique(labels):
        tmp = embeddings[labels==label]
        plt.scatter(tmp[:, 0], tmp[:, 1], label=label)

    plt.legend()
    plt.show()

if __name__ == "__main__":
    
    from src.dataset import get_load_data

    _, val_set = get_load_data(dataset = "FashionMNIST")
    trained_model_path = "model_weights/model_weights.pt"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = torch.load(trained_model_path, map_location=torch.device(device))
    validation(model, val_set)
            