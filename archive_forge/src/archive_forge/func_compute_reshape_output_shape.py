import math
import numpy as np
import tree
from keras.src.api_export import keras_export
def compute_reshape_output_shape(input_shape, newshape, newshape_arg_name):
    """Converts `-1` in `newshape` to either an actual dimension or `None`.

    This utility does not special case the 0th dimension (batch size).
    """
    unknown_dim_count = newshape.count(-1)
    if unknown_dim_count > 1:
        raise ValueError(f'There must be at most one unknown dimension (-1) in {newshape_arg_name}. Received: {newshape_arg_name}={newshape}.')
    if None in input_shape:
        return tuple((dim if dim != -1 else None for dim in newshape))
    input_size = math.prod(input_shape)
    if unknown_dim_count == 0:
        if input_size != math.prod(newshape):
            raise ValueError(f'The total size of the tensor must be unchanged. Received: input_shape={input_shape}, {newshape_arg_name}={newshape}')
        return newshape
    known_output_size = 1
    unknown_dim_index = None
    for index, dim in enumerate(newshape):
        if dim == -1:
            unknown_dim_index = index
        else:
            known_output_size *= dim
    if known_output_size == 0 or input_size % known_output_size != 0:
        raise ValueError(f'The total size of the tensor must be unchanged, however, the input size cannot by divided by the specified dimensions in {newshape_arg_name}. Received: input_shape={input_shape}, {newshape_arg_name}={newshape}')
    output_shape = list(newshape)
    output_shape[unknown_dim_index] = input_size // known_output_size
    return tuple(output_shape)