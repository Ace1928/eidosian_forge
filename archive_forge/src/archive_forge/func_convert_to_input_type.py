import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework.weak_tensor import WeakTensor
def convert_to_input_type(base_input, input_type, dtype=None):
    if input_type == 'WeakTensor':
        return WeakTensor.from_tensor(constant_op.constant(base_input, dtype=dtype))
    elif input_type == 'Tensor':
        return constant_op.constant(base_input, dtype=dtype)
    elif input_type == 'NumPy':
        dtype = dtype.as_numpy_dtype if isinstance(dtype, dtypes.DType) else dtype
        return np.array(base_input, dtype=dtype)
    elif input_type == 'Python':
        return base_input
    else:
        raise ValueError(f'The provided input_type {input_type} is not supported.')