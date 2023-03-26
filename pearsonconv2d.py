@tf.function
def pearsoncoeff(windows_centered_flattened, weights_centered_flattened):
  """This function assumes weights and windows are already mean centered.
  windows_centered of shape (batch, n_windows_h, n_windows_w, ws1*ws2*n_ch_in)
  weights_centered of shape (window_size1 * window_size2 * n_ch_in, n_ch_out)
  Outputs are 'flat' pearson coefficients"""
  # Numerator is the dot product of each window and the kernel, i.e. sum of products of same-index values
  # i.e. matmul with result (n_windows, n_ch_out)
  numerator = tf.einsum("ijkh,hl->ijkl", windows_centered_flattened, weights_centered_flattened)
  # Denominator is the product of l2-norms (euclidean norms, vector magnitudes)
  # print(weights_centered_flattened.shape)
  # windows_centered_flattened = tf.reshape(windows_centered_flattened, shape=(windows_centered_flattened.shape[0],
  #                                                                            ))
  denominator = tf.math.multiply(tf.norm(windows_centered_flattened, axis=3, keepdims=True)+1e-12,
                                 tf.norm(weights_centered_flattened, axis=0, keepdims=True)+1e-12,
                                 )
  # denominator = tf.einsum("ijkh,hl->ijkl", windows_centered_flattened, weights_centered_flattened)

  pearson_coeff_flat = tf.math.divide(numerator, denominator)
  return pearson_coeff_flat

@tf.function
def sample_pearsoncoeff(x_sample, kernel, padding_val="SAME", strides=(1,1), rates=(1,1)):
  """This function takes a 2-D matrix (a single channel from a tensor (1xHxWxC)), 
  divides it in windows that are flatten, and mean-centers them.
  Kernel weights are assumed already mean centered.
  Outputs are 2-D matrix of the pearson coefficients between windows and kernel"""
  if padding_val=="same":
    padding_val=="SAME"
  elif padding_val=="valid":
    padding_val=="VALID"

  stride1, stride2 = strides
  rate1, rate2 = rates
  
  window_size1 = kernel.get_shape().as_list()[0]
  window_size2 = kernel.get_shape().as_list()[1]
  n_ch_in = kernel.get_shape().as_list()[2]
  n_ch_out = kernel.get_shape().as_list()[3]
  weights_flattened = tf.reshape(kernel, (window_size1*window_size2*n_ch_in, n_ch_out))
  
  # Divide input in flat windows
  x_windows = tf.image.extract_patches(x_sample, [1, window_size1, window_size2, 1], strides=[1, stride1, stride2, 1], rates=[1, rate1, rate2, 1], padding=padding_val)
  # x_windows of shape (batch_size, index_hs, index_ws, window_size1*window_size2*n_ch_in)

  # Mean-center windows
  x_windows_means = tf.reduce_mean(x_windows, axis=[3], keepdims=True)
  x_windows_centered = tf.math.subtract(x_windows, x_windows_means)
  # print(x_windows_centered.shape)
  # Compute correlations between windows and kernel that apply
  # x_windows_centered_flattened = tf.reshape(x_windows_centered, 
  #                                           shape=[x_windows_centered.shape[0], 
  #                                                  x_windows_centered.shape[1]*x_windows_centered.shape[2], x_windows_centered.shape[3]])
  y_windows_pearson = pearsoncoeff(x_windows_centered, weights_flattened) # shape (n_windows, n_ch_out)

  # Reshape correlations with abt (x.shape[0], ~x.shape[1], ~x.shape[2], x.shape[3])
  y_shape1 = x_windows.get_shape().as_list()[1]
  y_shape2 = x_windows.get_shape().as_list()[2]
  y_pearson = tf.reshape(y_windows_pearson, [-1, y_shape1, y_shape2, n_ch_out])
  return y_pearson

class StandardizedConv2DWithCall(tf.keras.layers.Conv2D):
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        # inputs.set_shape(shape=(batch_size, inputs.shape[1], inputs.shape[2], inputs.shape[3]))
        kernel_centered = self.kernel - tf.math.reduce_mean(self.kernel, axis=[0,1,2], keepdims=True)
        outputs = sample_pearsoncoeff(inputs, kernel_centered)
        # outputs.set_shape(shape=(batch_size, outputs.shape[1], outputs.shape[2], outputs.shape[3]))
        return outputs