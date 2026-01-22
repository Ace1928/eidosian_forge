import collections
import os
import re
from packaging import version
from tensorflow.compiler.tf2tensorrt import _pywrap_py_utils
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.framework import dtypes
def get_trtengineop_io_nodes_count(node, key):
    """Returns the number of input/output nodes of a TRTEngineOp."""
    return len(node.attr[key].list.type)