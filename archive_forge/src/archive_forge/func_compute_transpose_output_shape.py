import math
import numpy as np
import tree
from keras.src.api_export import keras_export
def compute_transpose_output_shape(input_shape, axes):
    """Compute the output shape for the `transpose` operation.

    Args:
        input_shape: Input shape.
        axes: Permutation of the dimensions for the `transpose` operation.

    Returns:
        Tuple of ints: The output shape after the `transpose` operation.
    """
    input_shape = list(input_shape)
    if axes is None:
        return tuple(input_shape[::-1])
    if len(axes) != len(input_shape):
        raise ValueError(f'axis must be a list of the same length as the input shape, expected {len(input_shape)}, but received {len(axes)}.')
    return tuple((input_shape[ax] for ax in axes))