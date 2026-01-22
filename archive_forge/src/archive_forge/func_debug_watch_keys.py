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
def debug_watch_keys(self, node_name, device_name=None):
    """Get all tensor watch keys of given node according to partition graphs.

    Args:
      node_name: (`str`) name of the node.
      device_name: (`str`) name of the device. If there is only one device or if
        node_name exists on only one device, this argument is optional.

    Returns:
      (`list` of `str`) all debug tensor watch keys. Returns an empty list if
        the node name does not correspond to any debug watch keys.

    Raises:
      `LookupError`: If debug watch information has not been loaded from
        partition graphs yet.
    """
    try:
        device_name = self._infer_device_name(device_name, node_name)
    except ValueError:
        return []
    if node_name not in self._debug_watches[device_name]:
        return []
    watch_keys = []
    for watched_slot in self._debug_watches[device_name][node_name]:
        debug_ops = self._debug_watches[device_name][node_name][watched_slot]
        for debug_op in debug_ops:
            watch_keys.append(_get_tensor_watch_key(node_name, watched_slot, debug_op))
    return watch_keys