import tensorflow as tf

@tf.function
def depthwise_samplewise_pearsonconv2d(x, W, strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding_val='SAME'):
    """
    Input x has shape (Height, Width, 1)
    Kernel W has shape (W_height, W_width)
    """
    W = tf.expand_dims(W, axis=-1)
    W = tf.expand_dims(W, axis=0)
    x = tf.expand_dims(x, axis=0)
    # Flatten kernel
    window_size1 = W.get_shape().as_list()[1]
    window_size2 = W.get_shape().as_list()[2]
    W_flat = tf.reshape(W, (1,1,1,window_size1*window_size2))
    # Center_kernel
    W_means =  tf.reduce_mean(W_flat, axis=[3], keepdims=True)
    W_centered = tf.math.subtract(W_flat, W_means)
    W_norm = tf.norm(W_centered, axis=3, keepdims=True) + 1e-12
    
    # Split the input tensor into windows
    windows = tf.image.extract_patches(x, [1, window_size1, window_size2, 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding=padding_val)
    
    # Compute the mean of each window
    window_means = tf.reduce_mean(windows, axis=[3], keepdims=True)

    # Subtract the mean from each window
    windows_centered = tf.math.subtract(windows, window_means) # the windows are already flattened, the means are broadcast
    
    # Compute the norms of centered windows
    window_norms = tf.norm(windows_centered, axis=3, keepdims=True) + 1e-12
    
    # Perform the dot product of centered quantities
    pearson = tf.reduce_sum(tf.math.multiply(windows_centered, W_centered), axis=3, keepdims=True)
    
    # Divide the result by the product of magnitudes
    denominator = tf.math.multiply(window_norms, W_norm)
    pearson = tf.math.divide(pearson, denominator)
    return pearson[0]

@tf.function
def samplewise_pearsonconv2d(x, W, strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding_val='SAME'):
    if padding_val=="same":
        padding_val = "SAME"
    if padding_val=="valid":
        padding_val = "VALID"
    # W.shape=(k_size, k_size, in, out)
    # for every output channel we are applying a filter (k_size, k_size, in, 1) whose output is sum_reduced along the input channel dim
    feature_maps = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    for channel_out in tf.range(W.get_shape().as_list()[3]):
        W_filter = W[:,:,:,channel_out]
        filter_response = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        for channel_in in tf.range(W.get_shape().as_list()[2]):
            W_slice = W_filter[:,:,channel_in]
            x_slice = tf.expand_dims(x[:,:,channel_in], -1)
            out_slice = depthwise_samplewise_pearsonconv2d(x_slice, W_slice, strides=strides, rates=rates, padding_val=padding_val)
            filter_response = filter_response.write(channel_in, out_slice)
        # end of inputchannel iterations, time to output single response for W_filter
        filter_response = tf.math.reduce_sum(filter_response.stack(), axis=0, keepdims=False)[:,:,0]
        # time to concatenate this response with others
        feature_maps = feature_maps.write(channel_out, filter_response)
    # end of outputchannel iterations, time to concatenate filter_responses into one long tensor
    feature_maps = feature_maps.stack()
    # (None, outsize1, outsize2, 1) needs to be reshaped
    feature_maps = tf.transpose(feature_maps, perm=[2,1,0])
    # feature_maps = tf.squeeze(feature_maps, axis=0)
    return feature_maps

@tf.function
def batch_pearsonconv2d(x, W, strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding_val="VALID"):
    batch_maps = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    for sample in tf.range(x.get_shape().as_list()[0]):
        # x_sample = tf.expand_dims(x[sample,:,:,:], 0)
        x_sample = x[sample,:,:,:]
        feat_maps = samplewise_pearsonconv2d(x_sample, W, strides=strides, rates=rates, padding_val=padding_val)
        # feat_maps = tf.expand_dims(feat_maps, -1)
        feat_maps = tf.expand_dims(feat_maps, 0)
        # feat_maps = tf.reshape(feat_maps, [feat_maps.get_shape().as_list()[0],feat_maps.get_shape().as_list()[1],1])
        batch_maps = batch_maps.write(sample, feat_maps)
    batch_maps = batch_maps.concat()
    return batch_maps