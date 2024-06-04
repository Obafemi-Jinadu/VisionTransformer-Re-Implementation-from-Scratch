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
from torch import nn, optim

class PatchEmbedding(nn.Module):
    def __init__(self, embed_dim, patch_size, input_size = 224,channels = 3):
        super().__init__()
        self.embed_dim = embed_dim
        self.channels = channels
        self.patch_size = patch_size 
        self.input_size = input_size
        #self.fc = nn.Linear(self.patch_size*self.patch_size*self.channels, self.embed_dim)
        self.convProjection = nn.Conv2d(self.channels,self.embed_dim,self.patch_size,self.patch_size)
        self.h_w = (((self.input_size - self.patch_size)//self.patch_size )+1)**2 # w - f + 2p/s +1 formula
        self.pos_embed = nn.Parameter(torch.randn(1, self.h_w+1, self.embed_dim))
        self.cls_embed = nn.Parameter(torch.randn(1, 1, self.embed_dim)) 
        
    def forward(self,x):
        x = self.convProjection(x)
        x = rearrange(x, 'b c h w ->b (h w) c')
        #self.pos_embed = nn.Parameter(torch.randn(1, x.shape[1]+1, x.shape[2]))
        #self.cls_embed = nn.Parameter(torch.randn(1, 1, x.shape[2])) 
        #print(self.cls_embed)
        cls_embed = self.cls_embed.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_embed,x),dim=1)
        x+=self.pos_embed
        
        return x
    
class AttentionBlock(nn.Module):
    def __init__(self, embeddings, head_size, bias=True):
        super().__init__()
            
        #every token emitts 2 vectors, key and query
        #query- what am I looking for
        #key what do I contain
        
        self.embeddings = embeddings
        self.head_size = head_size
        #self.key = nn.Linear(embeddings, head_size, bias =bias)
        #self.query = nn.Linear(embeddings, head_size, bias =bias)
        #self.value = nn.Linear(embeddings, head_size, bias =bias)
        self.qkv = nn.Linear(self.embeddings, self.head_size*3,bias=bias)
        
    def forward(self,x):
        #K = self.key(x)
        #Q = self.query(x)
        #V = self.value(x)
        x = self.qkv(x)
        B, N, emb = x.shape
        x = x.view(B, N, emb//3, 3)
        Q, K, V = torch.unbind(x,-1)
        wei = Q@K.transpose(-2,-1) #(B N C) (B C N) --> B N N
        wei/=self.head_size**0.5
        wei = F.softmax(wei, dim = -1)
        output = wei @ V
        #print('output', output.shape)
        return output
    
class MSA(nn.Module):
    def __init__(self, embeddings, n_heads):
        super().__init__()
        #self.n_heads = n_heads 
        #self.embeddings = embeddings
        self.D_h = embeddings//n_heads 
        self.heads = nn.ModuleList([AttentionBlock(embeddings, self.D_h) for i in range(n_heads)])
        self.total_head_embed = self.D_h*n_heads
        self.proj_back = nn.Linear(self.total_head_embed, embeddings)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):   
        x = torch.cat(([head(x) for head in self.heads]),dim = -1)
        #print('proj_back', x.shape)
        x = self.proj_back(x)
        x = self.dropout(x)
        return x
    

class MLP(nn.Module):
    def __init__(self, embeddings, mlp_ratio=4):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.embeddings = embeddings
        self.mlp_ratio = mlp_ratio
        self.fc = nn.Linear(self.embeddings, self.embeddings*self.mlp_ratio)
        self.gelu = nn.GELU()   
        self.fc2 = nn.Linear(self.embeddings*self.mlp_ratio, self.embeddings)
        
    def forward(self, x):
        x = self.fc(x)
        x = self.dropout(x)
        x = self.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
        
    
class TransformerEncoder(nn.Module):
    def __init__(self, embeddings,n_heads):
        super().__init__()
        self.embeddings = embeddings
        self.layernorm1 = nn.LayerNorm(self.embeddings)    
        self.n_heads = n_heads
        self.msa = MSA(self.embeddings, self.n_heads)
        self.mlp = MLP(self.embeddings)
        self.layernorm2 = nn.LayerNorm(self.embeddings)  
        
    def forward(self, x):
        x_skip = x
        x = self.layernorm1(x)
        x = self.msa(x)
        x+=x_skip
        x_skip2 = x
        x = self.layernorm2(x)
        x = self.mlp(x)+x_skip2
        return x
    
class ViTBackbone(nn.Module):
    def __init__(self, input_embeddings, patch_size, n_heads, layers, input_img=224, channels=3):
        super().__init__()
        self.embeddings = input_embeddings
        self.patch_size = patch_size
        self.input_img = input_img
        self.channels = channels
        self.n_heads = n_heads
        self.layers =layers
        self.patch_embed =  PatchEmbedding(self.embeddings,self.patch_size,self.input_img)
        self.encoders = nn.ModuleList([TransformerEncoder(self.embeddings, self.n_heads) for i in range(self.layers)])
          
    def forward(self, x):
        x = self.patch_embed(x)
        for layer in self.encoders:
            x = layer(x)
        return x
        


