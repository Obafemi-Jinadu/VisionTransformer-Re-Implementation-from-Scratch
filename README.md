# VisionTransformer-Re-implementation-from-Scratch
A Re-implementation of Vision Transformer from scratch based off of the paper 'AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE'


 <h1 align="center"><img src="https://github.com/Obafemi-Jinadu/VisionTransformer-Re-Implementation-from-Scratch/blob/5fd16a834708566c30b48b07a2faa79e099ea4c9/files/vitoverview.png" width=700/></h1>
  <h1 align="center" width=50>(a)</h1>

 <h1 align="center"><img src="https://github.com/Obafemi-Jinadu/VisionTransformer-Re-Implementation-from-Scratch/blob/2a2e44d4243832aa95cec3f83495ea7965fdb1ea/files/im2patches.png" /></h1>
 <h1 align="center" width=50>(b)</h1> Figure 1: a) Vision Transformer overwivew b) image to patches.



Images are broken into non-overlapping patches. Patches are projected to embeddings of higher dimensions using either
- A linear layer (3, embedding_dimension) or
- A 2d convolution operation (input_channel = 3, output_channel = embedding_dimension, kernel = patch_size, stride = patch_size).
  
Unlike convolutional neural networks, transformers do not have the ability to understand the spatial ordering of patches which is why an inducive bias is incorporated into the transformer encoding in the form of postion encodings. Think of it like including an index to ID every image patch but instead, for better results the position encoding is a learnable parameter.
