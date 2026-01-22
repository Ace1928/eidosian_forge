import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src.layers.pooling.base_global_pooling1d import GlobalPooling1D
from tensorflow.python.util.tf_export import keras_export
Global average pooling operation for temporal data.

    Examples:

    >>> input_shape = (2, 3, 4)
    >>> x = tf.random.normal(input_shape)
    >>> y = tf.keras.layers.GlobalAveragePooling1D()(x)
    >>> print(y.shape)
    (2, 4)

    Args:
      data_format: A string,
        one of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape
        `(batch, steps, features)` while `channels_first`
        corresponds to inputs with shape
        `(batch, features, steps)`.
      keepdims: A boolean, whether to keep the temporal dimension or not.
        If `keepdims` is `False` (default), the rank of the tensor is reduced
        for spatial dimensions.
        If `keepdims` is `True`, the temporal dimension are retained with
        length 1.
        The behavior is the same as for `tf.reduce_mean` or `np.mean`.

    Call arguments:
      inputs: A 3D tensor.
      mask: Binary tensor of shape `(batch_size, steps)` indicating whether
        a given step should be masked (excluded from the average).

    Input shape:
      - If `data_format='channels_last'`:
        3D tensor with shape:
        `(batch_size, steps, features)`
      - If `data_format='channels_first'`:
        3D tensor with shape:
        `(batch_size, features, steps)`

    Output shape:
      - If `keepdims`=False:
        2D tensor with shape `(batch_size, features)`.
      - If `keepdims`=True:
        - If `data_format='channels_last'`:
          3D tensor with shape `(batch_size, 1, features)`
        - If `data_format='channels_first'`:
          3D tensor with shape `(batch_size, features, 1)`
    