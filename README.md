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
- Encoder Backbone
  - Layer norms, Self-attention mechanism, MLP
- Model Head for computer vision downstream tasks like classification, pose estimation, segmentation etc.
- Discuss More findings

# References
[1] Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S. and Uszkoreit, J., 2020. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929.
