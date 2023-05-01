import torch.nn as nn
from src.model import Network
import torch.optim as optim
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchsummary import summary

def train(epochs, train_set, in_channels = 3, num_classes = 10, save_model_path = "model_weights/model_weights.pt"):

    loss_function = nn.TripletMarginWithDistanceLoss(distance_function=nn.PairwiseDistance())
    # network = ResNet50Above(in_channels = in_channels, num_classes = num_classes)
    network = Network(in_channels = in_channels)
    # summary(network, (in_channels,224,224))
    optimizer = optim.SGD(network.parameters(), lr=3e-4, weight_decay=5e-5)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # print (device)
    network.to(device)
    # start = time.time()
    # add tqdm here
    train_dataloader = DataLoader(train_set, batch_size=20)
    for epoch in range(epochs):
        print (f"Epoch {epoch + 1}:")
        # for i in tqdm(range(X.shape[0])):
        with tqdm(train_dataloader) as tepoch:
            for anchor, positive, negative, label in tepoch:
                # print (img.shape)
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
    from src.dataloader import TripletLossDataset
    train_set = TripletLossDataset(train=True)
    train(epochs = 3, train_set = train_set, in_channels = 1, num_classes = 10)

    # train_set, test_set = get_load_data(dataset = "FashionMNIST")
    # train(epochs = 1, train_set = train_set, in_channels = 1, num_classes = 10)
    