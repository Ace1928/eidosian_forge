import copy
import datetime
import sys
from absl import logging
import flatbuffers
from tensorflow.core.protobuf import config_pb2 as _config_pb2
from tensorflow.core.protobuf import meta_graph_pb2 as _meta_graph_pb2
from tensorflow.lite.python import conversion_metadata_schema_py_generated as conversion_metadata_fb
from tensorflow.lite.python import schema_py_generated as schema_fb
from tensorflow.lite.python import schema_util
from tensorflow.lite.python import tflite_keras_util as _tflite_keras_util
from tensorflow.lite.python.op_hint import convert_op_hints_to_stubs
from tensorflow.lite.python.op_hint import find_all_hinted_output_nodes
from tensorflow.lite.tools import flatbuffer_utils
from tensorflow.python.eager import function
from tensorflow.python.framework import convert_to_constants as _convert_to_constants
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import error_interpolation as _error_interpolation
from tensorflow.python.grappler import tf_optimizer
from tensorflow.python.training.saver import export_meta_graph as _export_meta_graph
def get_tensors_from_tensor_names(graph, tensor_names):
    """Gets the Tensors associated with the `tensor_names` in the provided graph.

  Args:
    graph: TensorFlow Graph.
    tensor_names: List of strings that represent names of tensors in the graph.

  Returns:
    A list of Tensor objects in the same order the names are provided.

  Raises:
    ValueError:
      tensor_names contains an invalid tensor name.
  """
    tensor_name_to_tensor = {}
    for op in graph.get_operations():
        for tensor in op.values():
            tensor_name_to_tensor[get_tensor_name(tensor)] = tensor
    tensors = []
    invalid_tensors = []
    for name in tensor_names:
        if not isinstance(name, str):
            raise ValueError("Invalid type for a tensor name in the provided graph. Expected type for a tensor name is 'str', instead got type '{}' for tensor name '{}'".format(type(name), name))
        tensor = tensor_name_to_tensor.get(name)
        if tensor is None:
            invalid_tensors.append(name)
        else:
            tensors.append(tensor)
    if invalid_tensors:
        raise ValueError("Invalid tensors '{}' were found.".format(','.join(invalid_tensors)))
    return tensors