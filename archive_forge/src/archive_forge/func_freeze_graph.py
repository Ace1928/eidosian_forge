import argparse
import re
import sys
from absl import app
from google.protobuf import text_format
from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf import saver_pb2
from tensorflow.core.protobuf.meta_graph_pb2 import MetaGraphDef
from tensorflow.python.checkpoint import checkpoint_management
from tensorflow.python.client import session
from tensorflow.python.framework import convert_to_constants
from tensorflow.python.framework import importer
from tensorflow.python.platform import gfile
from tensorflow.python.saved_model import loader
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.tools import saved_model_utils
from tensorflow.python.training import py_checkpoint_reader
from tensorflow.python.training import saver as saver_lib
def freeze_graph(input_graph, input_saver, input_binary, input_checkpoint, output_node_names, restore_op_name, filename_tensor_name, output_graph, clear_devices, initializer_nodes, variable_names_whitelist='', variable_names_denylist='', input_meta_graph=None, input_saved_model_dir=None, saved_model_tags=tag_constants.SERVING, checkpoint_version=saver_pb2.SaverDef.V2):
    """Converts all variables in a graph and checkpoint into constants.

  Args:
    input_graph: A `GraphDef` file to load.
    input_saver: A TensorFlow Saver file.
    input_binary: A Bool. True means input_graph is .pb, False indicates .pbtxt.
    input_checkpoint: The prefix of a V1 or V2 checkpoint, with V2 taking
      priority.  Typically the result of `Saver.save()` or that of
      `tf.train.latest_checkpoint()`, regardless of sharded/non-sharded or
      V1/V2.
    output_node_names: The name(s) of the output nodes, comma separated.
    restore_op_name: Unused.
    filename_tensor_name: Unused.
    output_graph: String where to write the frozen `GraphDef`.
    clear_devices: A Bool whether to remove device specifications.
    initializer_nodes: Comma separated list of initializer nodes to run before
                       freezing.
    variable_names_whitelist: The set of variable names to convert (optional, by
                              default, all variables are converted),
    variable_names_denylist: The set of variable names to omit converting
                              to constants (optional).
    input_meta_graph: A `MetaGraphDef` file to load (optional).
    input_saved_model_dir: Path to the dir with TensorFlow 'SavedModel' file and
                           variables (optional).
    saved_model_tags: Group of comma separated tag(s) of the MetaGraphDef to
                      load, in string format.
    checkpoint_version: Tensorflow variable file format (saver_pb2.SaverDef.V1
                        or saver_pb2.SaverDef.V2).
  Returns:
    String that is the location of frozen GraphDef.
  """
    input_graph_def = None
    if input_saved_model_dir:
        input_graph_def = saved_model_utils.get_meta_graph_def(input_saved_model_dir, saved_model_tags).graph_def
    elif input_graph:
        input_graph_def = _parse_input_graph_proto(input_graph, input_binary)
    input_meta_graph_def = None
    if input_meta_graph:
        input_meta_graph_def = _parse_input_meta_graph_proto(input_meta_graph, input_binary)
    input_saver_def = None
    if input_saver:
        input_saver_def = _parse_input_saver_proto(input_saver, input_binary)
    return freeze_graph_with_def_protos(input_graph_def, input_saver_def, input_checkpoint, output_node_names, restore_op_name, filename_tensor_name, output_graph, clear_devices, initializer_nodes, variable_names_whitelist, variable_names_denylist, input_meta_graph_def, input_saved_model_dir, [tag for tag in saved_model_tags.replace(' ', '').split(',') if tag], checkpoint_version=checkpoint_version)