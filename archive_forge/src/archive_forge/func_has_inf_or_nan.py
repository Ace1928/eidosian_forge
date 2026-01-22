import collections
import glob
import json
import os
import platform
import re
import numpy as np
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import types_pb2
from tensorflow.core.util import event_pb2
from tensorflow.python.debug.lib import debug_graphs
from tensorflow.python.framework import tensor_util
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat
def has_inf_or_nan(datum, tensor):
    """A predicate for whether a tensor consists of any bad numerical values.

  This predicate is common enough to merit definition in this module.
  Bad numerical values include `nan`s and `inf`s.
  The signature of this function follows the requirement of the method
  `DebugDumpDir.find()`.

  Args:
    datum: (`DebugTensorDatum`) Datum metadata.
    tensor: (`numpy.ndarray` or None) Value of the tensor. None represents
      an uninitialized tensor.

  Returns:
    (`bool`) True if and only if tensor consists of any nan or inf values.
  """
    _ = datum
    if isinstance(tensor, InconvertibleTensorProto):
        return False
    elif np.issubdtype(tensor.dtype, np.floating) or np.issubdtype(tensor.dtype, np.complexfloating) or np.issubdtype(tensor.dtype, np.integer):
        return np.any(np.isnan(tensor)) or np.any(np.isinf(tensor))
    else:
        return False