import collections
import functools
import itertools
from typing import List, Dict, Optional, Union
import numpy as np
from tensorflow.dtensor.proto import layout_pb2
from tensorflow.python import _pywrap_dtensor_device
from tensorflow.python.framework import device as tf_device
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.util.tf_export import tf_export
def coords(self, device_idx: int) -> tensor.Tensor:
    """Converts the device index into a tensor of mesh coordinates."""
    strides = ops.convert_to_tensor(self.strides)
    shape = ops.convert_to_tensor(self.shape())
    return device_idx // strides % shape