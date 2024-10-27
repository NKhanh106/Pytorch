import torch
import torchvision
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split

class MnistModel(nn.Module):
    #mang noron voi 1 lop
    def __init__(self, in_size, hidden_size, out_size):
        super().__init__()
        self.linear1 = nn.Linear(in_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, out_size)
        
    def forward(self, xb):
        xb = xb.view(xb.size(0), -1)
        out = self.linear1(xb)
        out = F.relu(out)
        out = self.linear2(out)
        return out
    
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)
        loss = F.cross_entropy(out, labels)
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        return {'val_loss': loss, 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean() 
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()  
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def get_default_device():
    #uu tien chon gpu, neu khong thi chay bang cpu
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')    

def to_device(data, device):
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        return len(self.dl)

def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    #train model dung gradient descent
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # train
        for batch in train_loader:
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # validation 
        result = evaluate(model, val_loader)
        model.epoch_end(epoch, result)
        history.append(result)
    return history

dataset = MNIST(root='data/', download=True, transform=ToTensor())
if __name__ == '__main__':
    val_size = 10000
    train_size = len(dataset) - val_size

    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    batch_size=128

    train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size*2, num_workers=4, pin_memory=True)

    for images, labels in train_loader:
        inputs = images.reshape(-1, 784)
        break

    input_size = inputs.shape[-1]
    hidden_size = 32

    layer1 = nn.Linear(input_size, hidden_size)
    layer1_outputs = layer1(inputs)
    layer1_outputs_direct = inputs @ layer1.weight.t() + layer1.bias
    torch.allclose(layer1_outputs, layer1_outputs_direct, 1e-3)

    relu_outputs = F.relu(layer1_outputs)
    output_size = 10
    layer2 = nn.Linear(hidden_size, output_size)
    layer2_outputs = layer2(relu_outputs)
    print(F.cross_entropy(layer2_outputs, labels))  #

    outputs = (F.relu(inputs @ layer1.weight.t() + layer1.bias)) @ layer2.weight.t() + layer2.bias

    outputs2 = (inputs @ layer1.weight.t() + layer1.bias) @ layer2.weight.t() + layer2.bias

    combined_layer = nn.Linear(input_size, output_size)
    combined_layer.weight.data = layer2.weight @ layer1.weight
    combined_layer.bias.data = layer1.bias @ layer2.weight.t() + layer2.bias

    outputs3 = inputs @ combined_layer.weight.t() + combined_layer.bias

    input_size = 784
    hidden_size = 32 
    num_classes = 10

    model = MnistModel(input_size, hidden_size=32, out_size=num_classes)

    device = get_default_device()

    for images, labels in train_loader:
        images = to_device(images, device)
        break

    train_loader = DeviceDataLoader(train_loader, device)
    val_loader = DeviceDataLoader(val_loader, device)

    #model chay bang gpu
    model = MnistModel(input_size, hidden_size=hidden_size, out_size=num_classes)
    to_device(model, device)
    history = [evaluate(model, val_loader)]
    history += fit(5, 0.5, model, train_loader, val_loader)
    history += fit(5, 0.1, model, train_loader, val_loader)
