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
def strip_graph_default_valued_attrs(meta_graph_def):
    """Strips default valued attributes for node defs in given MetaGraphDef.

  This method also sets `meta_info_def.stripped_default_attrs` in the given
  `MetaGraphDef` proto to True.

  Args:
    meta_graph_def: `MetaGraphDef` protocol buffer

  Returns:
    None.
  """
    op_name_to_function = {}
    for function_def in meta_graph_def.graph_def.library.function:
        op_name_to_function[function_def.signature.name] = function_def

    def _strip_node_default_valued_attrs(node_def):
        """Removes default valued attributes from a single node def."""
        if node_def.op in op_name_to_function:
            return
        op_def = op_def_registry.get(node_def.op)
        if op_def is None:
            return
        attrs_to_strip = set()
        for attr_name, attr_value in node_def.attr.items():
            if _is_default_attr_value(op_def, attr_name, attr_value):
                attrs_to_strip.add(attr_name)
        for attr in attrs_to_strip:
            del node_def.attr[attr]
    for node_def in meta_graph_def.graph_def.node:
        _strip_node_default_valued_attrs(node_def)
    for function_def in meta_graph_def.graph_def.library.function:
        for function_node_def in function_def.node_def:
            _strip_node_default_valued_attrs(function_node_def)
    meta_graph_def.meta_info_def.stripped_default_attrs = True