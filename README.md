# CosinePearsonLayers
This repository contains code to implement Cosine Similarity and Pearson Correlation "convolutions".
The main function is `sample_pearsoncoeff`, which can be used in custom layers inheriting from `tf.keras.layers.Conv2D()` to override the call.
This is done in `PearsonConv2D`.
`sample_pearsoncoeff` allows to compute the Pearson Correlation (i.e. Cosine Similarity between mean-centered vectors),
between a window from the input and the kernel.
The process automatically normalizes the intermediate output between $[-1,1]$, and then adds bias and passes through the activation function if specified.
Alternative implementations[1][2] are not inheriting the flexibility of `tf.keras.layers.Conv2D()`, or are not mean-centering the windows separately.
Implementation in [1] automatically implements a nonlinearity (exponent $p$ in the definition), and recommends using MaxAbsPooling, which is equivalent to MaxPooling after mirroring the negative side of the activation above the x axis.

# Background
Luo, Zhang, Yang on "Cosine Normalization" https://arxiv.org/abs/1702.05870.
# Motivation
Convolution and cross-correlation do not operate as commonly narrated, i.e. by finding pixel patterns similar to the filters.
# Objective
Finding a better suited operation to measure similarity between input patches and filters.
# Methods
Cosine similarity and Pearson correlation between a patch and a filter.
# Results
Let's compute and see!

# References
[1] https://github.com/brohrer/sharpened-cosine-similarity
[2] https://github.com/iwyoo/tf_conv_cosnorm
