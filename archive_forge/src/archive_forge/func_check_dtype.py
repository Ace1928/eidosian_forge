import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.module import module
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables as variables_module
from tensorflow.python.util import nest
def check_dtype(arg, dtype):
    """Check that arg.dtype == self.dtype."""
    if arg.dtype.base_dtype != dtype:
        raise TypeError(f'Expected argument to have dtype {dtype}. Found: {arg.dtype} in tensor {arg}.')