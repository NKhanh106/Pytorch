import os
import torch
import torchvision
import tarfile
import matplotlib
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets.utils import download_url
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor

def apply_kernel(image, kernel):
    ri, ci = image.shape
    rk, ck = kernel.shape
    ro, co = ri - rk + 1, ci - ck + 1
    output = torch.zeros([ro,co])
    for i in range(ro):
        for j in range(co):
            output[i, j] = torch.sum(image[i : i + rk, j : j + ck] * kernel)
    return output

def accuracy(outputs, labels):
    _, preds = torch.max(outputs,dim = 1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class ImageClassificationBase(nn.Module):
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
        return {'val_loss': loss.detach(), 'val_acc' : acc}
    
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return {'val_loss': epoch_loss.item(), 'val_acc' : epoch_acc.item()}

class ThisnnModel(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            #in: 3 x 32 x 32
            #relu :a < 0 : = 0,a > 0 : = a 
            nn.Conv2d(3, 32, kernel_size= 3, padding= 1),# 32 x 32 x 32
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size= 3, stride= 1, padding= 1),# 64 x 32 x 32
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 64 x 16 x 16

            nn.Conv2d(64, 128, kernel_size=3, stride= 1, padding= 1),# 128 x 16 x 16
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size= 3, stride= 1, padding= 1),# 128 x 16 x 16
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 128 x 8 x 8

            nn.Conv2d(128, 256, kernel_size= 3, stride= 1, padding= 1), # 256 x 8 x 8
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size= 3, stride= 1, padding= 1), # 256 x 8 x 8
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 256 x 4 x 4

            nn.Flatten(),
            nn.Linear(4 * 4 * 256, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
    def forward(self, xb):
        return self.network(xb)   

def get_default_device():
    #neu co gpu thi dung gpu, khong thi dung cpu
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking = True)

class DeviceDataLoader():
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
    
    def __iter__(self):
        for b in self.dl:
            yield to_device(b, self.device)
    
    def __len__(self):
        return len(self.dl)

@torch.no_grad
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func = torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        #train
        model.train()
        train_losses = []
        for batch in train_loader:
            loss = model.trainng_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        #xac thuc
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
    return history

dataset_url = "https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz"
download_url(dataset_url, '.')

with tarfile.open('./cifar10.tgz', 'r:gz') as tar:
    tar.extractall(path = './data')

classes = os.listdir('./data/cifar10/train')
dataset = ImageFolder('./data/cifar10/train', transform=ToTensor())

matplotlib.rcParams['figure.facecolor'] = '#ffffff'

random_seed = 36
torch.manual_seed(random_seed)

val_size = 5000 #tap xac thuc
train_size = len(dataset) - val_size #tap train = 50000 - 5000
train_ds, val_ds = random_split(dataset, [train_size, val_size]) # chia dataset thanh 2 phan la validation va train
batch_size = 128

train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers= 4, pin_memory=True)
val_dl = DataLoader(val_ds, batch_size * 2, num_workers= 4, pin_memory= True)

simple_model = nn.Sequential(
    nn.Conv2d(3, 8, kernel_size= 3, stride= 1, padding= 1),
    nn.MaxPool2d(2, 2)
)

model = ThisnnModel()

device = get_default_device()

traindl = DeviceDataLoader(train_dl, device)
val_dl = DeviceDataLoader(val_dl, device)
to_device(model, device)

