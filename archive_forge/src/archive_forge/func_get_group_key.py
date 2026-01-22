import copy
import threading
from typing import Callable, List, Optional, Union
from tensorflow.python.distribute import collective_util
from tensorflow.python.distribute import values as value_lib
from tensorflow.python.eager import backprop_util
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import collective_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nccl_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.types import core
def get_group_key(self, devices):
    """Returns a group key for the list of local devices.

    The same group key is returned if the list of local devices is the same.

    Args:
      devices: a list of local canonical device strings in a collective group.

    Returns:
      a group key.
    """
    with self._lock:
        devices_key = ','.join(devices)
        if devices_key not in self._known_groups:
            self._known_groups[devices_key] = self._get_new_group_key(devices)
        return self._known_groups[devices_key]