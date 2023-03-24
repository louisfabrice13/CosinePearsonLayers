# CosinePearsonLayers
This repository contains code to implement Cosine Similarity and Pearson Correlation "convolutions".
The main function is `omissis` which can be used in custom layers inheriting from `tf.keras.layers.Conv2D()` to override the call.
This functions allows to compute the Cosine Similarity or Pearson Correlation (i.e. Cosine Similarity between mean-centered vectors),
between a window from the input and the kernel.
Alternative implementations are not inheriting the flexibility of `tf.keras.layers.Conv2D()`, or are not mean-centering the windows separately.

Convolutional Neural Networks perform "convolutions" that are often visualized as kernels sliding over the input height and width dimensions.
Mathematically, they instead perform convolutions without mirroring, i. e. cross-correlations between the input matrix and the kernel.
Conceptually, we see them as searching the input for windows that are similar to the kernel, which represent 
