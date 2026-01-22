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
def node_recipients(self, node_name, is_control=False, device_name=None):
    """Get recipient of the given node's output according to partition graphs.

    Args:
      node_name: (`str`) name of the node.
      is_control: (`bool`) whether control outputs, rather than non-control
        outputs, are to be returned.
      device_name: (`str`) name of the device. If there is only one device or if
        node_name exists on only one device, this argument is optional.

    Returns:
      (`list` of `str`) all inputs to the node, as a list of node names.

    Raises:
      LookupError: If node inputs and control inputs have not been loaded
         from partition graphs yet.
    """
    if not self._debug_graphs:
        raise LookupError('Node recipients are not loaded from partition graphs yet.')
    device_name = self._infer_device_name(device_name, node_name)
    debug_graph = self._debug_graphs[device_name]
    if is_control:
        return debug_graph.node_ctrl_recipients[node_name]
    else:
        return debug_graph.node_recipients[node_name]