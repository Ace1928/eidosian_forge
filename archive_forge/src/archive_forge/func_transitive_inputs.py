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
def transitive_inputs(self, node_name, include_control=True, include_reversed_ref=False, device_name=None):
    """Get the transitive inputs of given node according to partition graphs.

    Args:
      node_name: Name of the node.
      include_control: Include control inputs (True by default).
      include_reversed_ref: Whether a ref input, say from A to B, is to be also
        considered as an input from B to A. The rationale is that ref inputs
        generally let the recipient (e.g., B in this case) mutate the value of
        the source (e.g., A in this case). So the reverse direction of the ref
        edge reflects the direction of information flow.
      device_name: (`str`) name of the device. If there is only one device or if
        node_name exists on only one device, this argument is optional.

    Returns:
      (`list` of `str`) all transitive inputs to the node, as a list of node
        names.

    Raises:
      LookupError: If node inputs and control inputs have not been loaded
         from partition graphs yet.
    """
    if not self._debug_graphs:
        raise LookupError('Node inputs are not loaded from partition graphs yet.')
    device_name = self._infer_device_name(device_name, node_name)
    input_lists = [self._debug_graphs[device_name].node_inputs]
    if include_control:
        input_lists.append(self._debug_graphs[device_name].node_ctrl_inputs)
    if include_reversed_ref:
        input_lists.append(self._debug_graphs[device_name].node_reversed_ref_inputs)
    tracer = debug_graphs.DFSGraphTracer(input_lists, skip_node_names=self._get_merge_node_names(device_name))
    tracer.trace(node_name)
    return tracer.inputs()