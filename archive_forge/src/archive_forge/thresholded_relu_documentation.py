import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src.engine.base_layer import Layer
from keras.src.utils import tf_utils
from tensorflow.python.util.tf_export import keras_export
Thresholded Rectified Linear Unit.

    It follows:

    ```
        f(x) = x for x > theta
        f(x) = 0 otherwise`
    ```

    Input shape:
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    Output shape:
        Same shape as the input.

    Args:
        theta: Float >= 0. Threshold location of activation.
    