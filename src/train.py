import torch.nn as nn
from src.model import BaseNetwork, SiameseNetwork
import torch.optim as optim
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchsummary import summary

def train(epochs, train_set, network = "Siamese", save_model_path = "model_weights/model_weights.pt"):

    loss_function = nn.TripletMarginWithDistanceLoss(distance_function=nn.PairwiseDistance())
    if network == "Siamese":
        network = SiameseNetwork(hw = 128)
    elif network == "Base":
        network = BaseNetwork()
    # summary(network, (in_channels,224,224))

    optimizer = optim.SGD(network.parameters(), lr=3e-4, weight_decay=5e-5)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    network.to(device)
    train_dataloader = DataLoader(train_set, batch_size=20)

    for epoch in range(epochs):
        print (f"Epoch {epoch + 1}:")
        with tqdm(train_dataloader) as tepoch:
            for anchor, positive, negative, label in tepoch:
                optimizer.zero_grad() 
                anchor_out = network.forward(anchor.to(device))
                positive_out = network.forward(positive.to(device))
                negative_out = network.forward(negative.to(device))
                loss = loss_function(anchor_out, positive_out, negative_out)
                loss.backward()
                optimizer.step()
                tepoch.set_postfix(loss=loss.item())
        
    print("training done")
    torch.save(network, save_model_path)

    return network

if __name__ == "__main__":

    torch.manual_seed(42)

    from src.dataset import get_load_data
    from src.dataloader import TripletLossDataset, generate_train_set
    
    # train_set = TripletLossDataset(train=True)
    # train(epochs = 10, train_set = train_set, network = "Base", in_channels = 1, num_classes = 10)

    train_set, test_set = get_load_data(dataset = "Flowers102")
    train_set = generate_train_set(train_set)
    triplet_loss_dataset = TripletLossDataset(train_set = train_set, train=True)
    train(epochs = 10, train_set = triplet_loss_dataset)
    