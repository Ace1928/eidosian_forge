import collections
import math
from tensorflow.python.framework import device as device_lib
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nccl_ops
def build_nccl_then_recursive_hd(input_tensors, red_op, un_op=None):
    """Construct hybrid of NCCL within workers, Recursive-HD across workers."""
    upper_level_f = lambda x: build_recursive_hd_all_reduce(x, red_op, un_op)
    return _build_nccl_hybrid(input_tensors, red_op, upper_level_f)