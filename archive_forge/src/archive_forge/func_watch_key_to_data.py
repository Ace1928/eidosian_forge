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
def watch_key_to_data(self, debug_watch_key, device_name=None):
    """Get all `DebugTensorDatum` instances corresponding to a debug watch key.

    Args:
      debug_watch_key: (`str`) debug watch key.
      device_name: (`str`) name of the device. If there is only one device or if
        the specified debug_watch_key exists on only one device, this argument
        is optional.

    Returns:
      A list of `DebugTensorDatum` instances that correspond to the debug watch
      key. If the watch key does not exist, returns an empty list.

    Raises:
      ValueError: If there are multiple devices that have the debug_watch_key,
        but device_name is not specified.
    """
    if device_name is None:
        matching_device_names = [name for name in self._watch_key_to_datum if debug_watch_key in self._watch_key_to_datum[name]]
        if not matching_device_names:
            return []
        elif len(matching_device_names) == 1:
            device_name = matching_device_names[0]
        else:
            raise ValueError("The debug watch key '%s' exists on multiple (%d) devices, but device name is not specified." % (debug_watch_key, len(matching_device_names)))
    elif device_name not in self._debug_key_to_datum:
        raise ValueError("There is no device named '%s' consisting of debug watch keys." % device_name)
    return self._watch_key_to_datum[device_name].get(debug_watch_key, [])