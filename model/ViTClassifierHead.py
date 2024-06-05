import torch
import torchvision
from ViTBackbone import ViTBackbone
import argparse
from torch import nn, optim
import einops
from einops import rearrange, reduce, repeat
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy
from torchvision import datasets, transforms, models
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchinfo import summary
import torchvision.transforms as T
import numpy as np
from torch.nn.functional import relu
import tqdm
from tqdm import tqdm
import math
from torchvision import datasets, transforms, models
from pathlib import Path

parser = argparse.ArgumentParser(description='Get arguments, data paths in this case.')
parser.add_argument('--data_path', type=str,
                    help='path to dataset')
parser.add_argument('--img_size',default=32, type=int,
                    help='image size pass as int')
parser.add_argument('--input_embeddings',default=768, type=int,
                    help='input_embeddings')
parser.add_argument('--patch_size',default=16, type=int,
                    help='patch size')
parser.add_argument('--num_classes',default=10, type=int,
                    help='num classes')
parser.add_argument('--batch_size',default=64, type=int,
                    help='batch size')
parser.add_argument('--n_heads',default=4, type=int,
                    help='n_heads')

parser.add_argument('--layers',default=12, type=int,
                    help='layers')
                    
parser.add_argument('--epochs',default=100, type=int,
                    help='training epochs')


args = parser.parse_args()

input_embeddings = args.input_embeddings
img_size = args.img_size
batch_size = args.batch_size
patch_size = args.patch_size
num_classes = args.num_classes
n_heads = args.n_heads
layers = args.layers
epochs = args.epochs
data_path = Path(args.data_path)

class ViTClassifier(nn.Module):
    def __init__(self, input_embeddings, patch_size, n_heads,layers, num_classes, input_img=img_size, channels=3):
        super().__init__()
        self.embeddings = input_embeddings
        self.patch_size = patch_size
        self.input_img = input_img
        self.channels = channels
        self.n_heads = n_heads
        self.layers =layers
        self.num_classes = num_classes
        self.layernorm = nn.LayerNorm(self.embeddings)
        self.backbone = ViTBackbone(self.embeddings, self.patch_size, self.n_heads, self.layers, self.input_img)
        self.hidden = nn.Linear(self.embeddings, self.embeddings//4)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.1)
        self.final_class = nn.Linear(self.embeddings//4,self.num_classes)
        
    def forward(self, x):
        x = self.backbone(x)
        x = x[:,0,:]
        x = self.layernorm(x)
        x = self.hidden(x)
        x = self.tanh(x)
        x = self.dropout(x)
        x = self.final_class(x)    
        return x 
        
if __name__ == "__main__":
	model = ViTClassifier(input_embeddings, patch_size, n_heads,layers, num_classes, input_img=img_size)
	
	data_transforms = transforms.Compose([
                                        transforms.Resize((img_size,img_size)),
                                       transforms.RandomRotation(30),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
                                       ])

	test_transforms = transforms.Compose([
		                                transforms.Resize((img_size,img_size)),
		                               transforms.ToTensor(),
		                               transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
		                               ])


	
	train_data = datasets.ImageFolder(data_path, transform=data_transforms)
	trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
	
	
	
	
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	model = model.to(device)
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.AdamW(model.parameters(), lr = 0.001,weight_decay=0.01)
	#optimizer = optim.Adam(model.parameters(), lr = 0.001)
	#num_steps = len(dataloader) * num_epochs
	#lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)
	#warmup_scheduler = warmup.UntunedLinearWarmup()
	train_loss_list = []
	test_loss_list = []
	acc_list = []
	for epoch in range(epochs):
         
         train_loss = 0
         
         for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward() 
            optimizer.step() 
            train_loss += loss.item() 
	    
         print(f' epoch : {epoch} | train loss:{train_loss/len(trainloader)}')
         torch.save(model.state_dict(), 'weights.pth')
		
