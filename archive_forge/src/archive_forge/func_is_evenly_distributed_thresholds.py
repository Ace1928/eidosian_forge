from enum import Enum
import functools
import weakref
import numpy as np
from tensorflow.python.compat import compat
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.keras import backend
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.utils.generic_utils import to_list
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variables as variables_module
from tensorflow.python.ops import weights_broadcast_ops
from tensorflow.python.ops.parallel_for import control_flow_ops as parallel_control_flow_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.util import tf_decorator
def is_evenly_distributed_thresholds(thresholds):
    """Check if the thresholds list is evenly distributed.

  We could leverage evenly distributed thresholds to use less memory when
  calculate metrcis like AUC where each individual threshold need to be
  evaluted.

  Args:
    thresholds: A python list or tuple, or 1D numpy array whose value is ranged
      in [0, 1].

  Returns:
    boolean, whether the values in the inputs are evenly distributed.
  """
    num_thresholds = len(thresholds)
    if num_thresholds < 3:
        return False
    even_thresholds = np.arange(num_thresholds, dtype=np.float32) / (num_thresholds - 1)
    return np.allclose(thresholds, even_thresholds, atol=backend.epsilon())