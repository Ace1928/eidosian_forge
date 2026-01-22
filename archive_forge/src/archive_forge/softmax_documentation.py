import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src.engine.base_layer import Layer
from keras.src.utils import tf_utils
from tensorflow.python.util.tf_export import keras_export
Softmax activation function.

    Example without mask:

    >>> inp = np.asarray([[1., 2., 1.]])
    >>> layer = tf.keras.layers.Softmax()
    >>> layer(inp).numpy()
    array([[0.21194157, 0.5761169 , 0.21194157]], dtype=float32)
    >>> mask = np.asarray([[True, False, True]], dtype=bool)
    >>> layer(inp, mask).numpy()
    array([[0.5, 0. , 0.5]], dtype=float32)

    Input shape:
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    Output shape:
        Same shape as the input.

    Args:
        axis: Integer, or list of Integers, axis along which the softmax
            normalization is applied.
    Call arguments:
        inputs: The inputs, or logits to the softmax layer.
        mask: A boolean mask of the same shape as `inputs`. The mask
            specifies 1 to keep and 0 to mask. Defaults to `None`.


    Returns:
        Softmaxed output with the same shape as `inputs`.
    