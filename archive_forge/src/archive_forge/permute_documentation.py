import copy
import tensorflow.compat.v2 as tf
from keras.src.engine.base_layer import Layer
from keras.src.engine.input_spec import InputSpec
from tensorflow.python.util.tf_export import keras_export
Permutes the dimensions of the input according to a given pattern.

    Useful e.g. connecting RNNs and convnets.

    Example:

    ```python
    model = Sequential()
    model.add(Permute((2, 1), input_shape=(10, 64)))
    # now: model.output_shape == (None, 64, 10)
    # note: `None` is the batch dimension
    ```

    Args:
      dims: Tuple of integers. Permutation pattern does not include the
        samples dimension. Indexing starts at 1.
        For instance, `(2, 1)` permutes the first and second dimensions
        of the input.

    Input shape:
      Arbitrary. Use the keyword argument `input_shape`
      (tuple of integers, does not include the samples axis)
      when using this layer as the first layer in a model.

    Output shape:
      Same as the input shape, but with the dimensions re-ordered according
      to the specified pattern.
    