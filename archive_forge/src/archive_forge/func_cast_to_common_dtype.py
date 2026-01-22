from keras.src import backend
from keras.src import ops
def cast_to_common_dtype(tensors):
    """Cast a list of tensors to a common dtype.

    If any tensor is floating-point, they will all be casted to the most-precise
    floating-point dtype. Otherwise the tensors are not casted.

    Args:
        tensors: A list of tensors.

    Returns:
        Same list, casted to a common dtype.
    """
    highest_float = None
    highest_float_size = -1
    for x in tensors:
        dtype = backend.standardize_dtype(x.dtype)
        if is_float(dtype):
            if highest_float is None or dtype_size(dtype) > highest_float_size:
                highest_float = dtype
                highest_float_size = dtype_size(dtype)
            elif dtype == 'float16' and highest_float == 'bfloat16':
                highest_float = 'float32'
                highest_float_size = dtype_size(highest_float)
    if highest_float:
        tensors = [ops.cast(x, highest_float) for x in tensors]
    return tensors