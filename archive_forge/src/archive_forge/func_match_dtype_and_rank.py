import copy
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.keras import losses as losses_mod
from tensorflow.python.keras import metrics as metrics_mod
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util import nest
def match_dtype_and_rank(y_t, y_p, sw):
    """Match dtype and rank of predictions."""
    if y_t.shape.rank == 1 and y_p.shape.rank == 2:
        y_t = array_ops.expand_dims_v2(y_t, axis=-1)
    if sw is not None:
        if sw.shape.rank == 1 and y_p.shape.rank == 2:
            sw = array_ops.expand_dims_v2(sw, axis=-1)
    if y_t.dtype.is_floating and y_p.dtype.is_floating or (y_t.dtype.is_integer and y_p.dtype.is_integer):
        y_t = math_ops.cast(y_t, y_p.dtype)
    if sw is not None:
        sw = math_ops.cast(sw, y_p.dtype)
    return (y_t, y_p, sw)