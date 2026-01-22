import copy
from packaging import version as packaging_version  # pylint: disable=g-bad-import-order
import os.path
import re
import sys
from google.protobuf.any_pb2 import Any
from google.protobuf import text_format
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import op_def_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.client import pywrap_tf_session as c_api
from tensorflow.python.eager import context
from tensorflow.python.framework import byte_swap_tensor as bst
from tensorflow.python.framework import error_interpolation
from tensorflow.python.framework import graph_io
from tensorflow.python.framework import importer
from tensorflow.python.framework import op_def_registry
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import versions
from tensorflow.python.lib.io import file_io
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat
def ops_used_by_graph_def(graph_def):
    """Collect the list of ops used by a graph.

  Does not validate that the ops are all registered.

  Args:
    graph_def: A `GraphDef` proto, as from `graph.as_graph_def()`.

  Returns:
    A list of strings, each naming an op used by the graph.
  """
    name_to_function = {}
    for fun in graph_def.library.function:
        name_to_function[fun.signature.name] = fun
    used_ops = set()
    functions_to_process = []

    def mark_op_as_used(op):
        if op not in used_ops and op in name_to_function:
            functions_to_process.append(name_to_function[op])
        used_ops.add(op)

    def process_node(node):
        mark_op_as_used(node.op)
        if node.op in ['PartitionedCall', 'StatefulPartitionedCall']:
            mark_op_as_used(node.attr['f'].func.name)
    for node in graph_def.node:
        process_node(node)
    while functions_to_process:
        fun = functions_to_process.pop()
        for node in fun.node_def:
            process_node(node)
    return [op for op in used_ops if op not in name_to_function]