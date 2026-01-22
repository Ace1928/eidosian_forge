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
def load_tensor_from_event(event):
    """Load a tensor from an Event proto.

  Args:
    event: The Event proto, assumed to hold a tensor value in its
        summary.value[0] field.

  Returns:
    The tensor value loaded from the event file, as a `numpy.ndarray`, if
    representation of the tensor value by a `numpy.ndarray` is possible.
    For uninitialized Tensors, returns `None`. For Tensors of data types that
    cannot be represented as `numpy.ndarray` (e.g., `tf.resource`), return
    the `TensorProto` protobuf object without converting it to a
    `numpy.ndarray`.
  """
    tensor_proto = event.summary.value[0].tensor
    shape = tensor_util.TensorShapeProtoToList(tensor_proto.tensor_shape)
    num_elements = 1
    for shape_dim in shape:
        num_elements *= shape_dim
    if tensor_proto.tensor_content or tensor_proto.string_val or (not num_elements):
        if tensor_proto.dtype == types_pb2.DT_RESOURCE:
            tensor_value = InconvertibleTensorProto(tensor_proto)
        else:
            try:
                tensor_value = tensor_util.MakeNdarray(tensor_proto)
            except KeyError:
                tensor_value = InconvertibleTensorProto(tensor_proto)
    else:
        tensor_value = InconvertibleTensorProto(tensor_proto, False)
    return tensor_value