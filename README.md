# VisionTransformer-Re-implementation-from-Scratch
A Re-implementation of Vision Transformer from scratch based off of the paper 'An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale'

# Vision Transformer (ViT) Overview - image from paper
 <h1 align="center"><img src="https://github.com/Obafemi-Jinadu/VisionTransformer-Re-Implementation-from-Scratch/blob/5fd16a834708566c30b48b07a2faa79e099ea4c9/files/vitoverview.png" width=700/></h1>

# Images to Patches
 <h1 align="center"><img src="https://github.com/Obafemi-Jinadu/VisionTransformer-Re-Implementation-from-Scratch/blob/2a2e44d4243832aa95cec3f83495ea7965fdb1ea/files/im2patches.png" /></h1>

# Patches Embeddings
Images are broken into non-overlapping patches. Patches are projected to embeddings of higher dimensions using either
- A linear layer (3, embedding_dimension) or
- A 2d convolution operation (input_channel = 3, output_channel = embedding_dimension, kernel = patch_size, stride = patch_size).
  
Unlike convolutional neural networks, transformers do not have the ability to understand the spatial ordering of patches which is why an inducive bias is incorporated into the transformer encoder in the form of postion encodings. Think of it like including an index to ID every image patch but instead, for better results the position encodings are learnable parameter.
The code implementation is shown below:
```
class PatchEmbedding(nn.Module):
    def __init__(self, embed_dim, patch_size, input_size = 224,channels = 3):
        super().__init__()
        self.embed_dim = embed_dim
        self.channels = channels
        self.patch_size = patch_size 
        self.input_size = input_size
        self.convProjection = nn.Conv2d(self.channels,self.embed_dim,self.patch_size,self.patch_size)
        self.h_w = (((self.input_size - self.patch_size)//self.patch_size )+1)**2 # w - f + 2p/s +1 formula
        self.pos_embed = nn.Parameter(torch.randn(1, self.h_w+1, self.embed_dim))
        self.cls_embed = nn.Parameter(torch.randn(1, 1, self.embed_dim)) 
        
    def forward(self,x):
        x = self.convProjection(x)
        x = rearrange(x, 'b c h w ->b (h w) c')
        cls_embed = self.cls_embed.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_embed,x),dim=1)
        x+=self.pos_embed  
        return x
```

# TODO - Discussions
# Transformer Encoder
The patch embeddings are passed to the transformer encoder block which typically contains layer normalizations, multi-head self attention, MLP layers.
## Layer Normalization
As is customary in machine learning tasks we want to restrict/clip input values within a range for efficient training and also prevent exploding and vanishing gradients. Batch Normalization helps us achieve, where normalization is carried out per data batch. However, this has its limitations. Consider a scenario where the batch size is too small. This would introduce a lot of noise as the mean and normalization of the small batch do not accurately represent the data distribution. On the other hand, when we have a training set that's too large, mini-batches could be split across different GPUs making the global normalization of said mini-batch inefficient as the GPUs involved would need to synchronize batch statistics. This was perfectly put in my go-to deep learning book ["Deep Learning Foundations and Concepts"](https://www.bishopbook.com/) by Christoper Bishop. Layer normalization does not have these shortcomings as normalization is carried out by layer making it independent of the batch size. Training deep transformers typically requires a lot of training data and often large batch sizes making layer normalization an ideal candidate.

## Self Attention Mechanism
The self-attention mechanism aids in extracting contextual information by considering the relationships between different parts of the input. For images, this involves breaking the image into patches. For instance, in a passport photo, which contains both background and foreground elements, one would expect low activation (relevance) between a patch that is purely background and a patch that is purely foreground. In self-attention, each patch embedding emits three vectors: a query, a key, and a value. The query represents the feature in question, while the keys are the features it seeks to match with. The matching is performed using a dot product operation, resulting in attention scores (activations). These scores act as scaling factors. A matrix multiplication is then performed between the attention scores and the value vectors (value vectors encode the input features). This process allows the model to combine and contextualize information from different parts of the image.

```
class AttentionBlock(nn.Module):
    def __init__(self, embeddings, head_size, bias=True):
        super().__init__()
        #every token emitts 2 vectors, key and query
        #query- what am I looking for
        #key what do I contain
        self.embeddings = embeddings
        self.head_size = head_size
        self.qkv = nn.Linear(self.embeddings, self.head_size*3,bias=bias)
        
    def forward(self,x):
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
```

## Multi-head Self Attention (MSA)
This mechanism involves running multiple self-attention operations (referred to as "heads") in parallel. Each attention head operates on the same input but with different learned parameters, allowing the model to capture different aspects of the contextual information. The outputs of these attention heads are then concatenated and projected through a linear layer. This process enables the model to aggregate and combine contextual information extracted from multiple perspectives, enhancing its ability to understand complex relationships within the data. Code snippet given below:
```
class MSA(nn.Module):
    def __init__(self, embeddings, n_heads):
        super().__init__()
        self.D_h = embeddings//n_heads 
        self.heads = nn.ModuleList([AttentionBlock(embeddings, self.D_h) for i in range(n_heads)])
        self.total_head_embed = self.D_h*n_heads
        self.proj_back = nn.Linear(self.total_head_embed, embeddings)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):   
        x = torch.cat(([head(x) for head in self.heads]),dim = -1)
        x = self.proj_back(x)
        x = self.dropout(x)
        return x
```

## MLP
This block takes in the output of the attention module expands and reduces the dimensionality to facilitate feature extraction and also introduces non-linearity to capture complete patterns. The code is given below:
```
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
```
Residual connections are also added to preserve and propagate information through the network and facilitate training of deeper networks. There's one between the input embeddings and the MSA and there is another between the output of the first residual block and the MLP output.

These parts all bundle up to make a transformer encder  block, there are typically several encoder blocks sequentially. As reported by the authors [1]: 
- The ViT-Base has 12 layers/blocks and 12 attention heads
- The ViT-Large has 24 layers/blocks and 16 attention heads
- The ViT-Huge has 32 layers/blocks and 16 attention heads

I call the combination of the input and transformer encoders the backbone and this backbone could be used for a suite of computer vision downstream tasks by attaching the appropriate head. 
- For example for a classifier like I used. It as simple as collecting the backbone passing to to some linear layers to gradual extract information and reduce dimensionality. The final layer should have output dimensions correspond to the number of classes e.g for cat, dog and horse classification the output layer must have 3 neurons.
The code is given below:
 ```
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
```
Recall: the class embedding was added to the input patches, that part is simply sliced out and fed to the linear layers for classification.

If I wanted to used it backbone for pose estimation task the final layer would simply be conv2D(64, number_keypoints) to generate heatmaps that have number of channels corresponding to the number of keypoints to be estimated. The backbone could be used for any downstream computer vision task that requires feature extraction.

The code can simply be run from terminal:
```
TODO
```

contact
```
obafemi.jinadu@tufts.edu
```

# References
[1] Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S. and Uszkoreit, J., 2020. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929.
