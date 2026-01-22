import argparse
import platform
import ast
import os
import re
from absl import app  # pylint: disable=unused-import
from absl import flags
from absl.flags import argparse_flags
import numpy as np
from tensorflow.core.example import example_pb2
from tensorflow.core.framework import types_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.debug.wrappers import local_cli_wrapper
from tensorflow.python.eager import def_function
from tensorflow.python.eager import function as defun
from tensorflow.python.framework import meta_graph as meta_graph_lib
from tensorflow.python.framework import ops as ops_lib
from tensorflow.python.framework import tensor_spec
from tensorflow.python.lib.io import file_io
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import load
from tensorflow.python.saved_model import loader
from tensorflow.python.saved_model import save
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.tools import saved_model_aot_compile
from tensorflow.python.tools import saved_model_utils
from tensorflow.python.tpu import tpu
from tensorflow.python.util.compat import collections_abc
def scan_meta_graph_def(meta_graph_def, op_denylist):
    """Scans meta_graph_def and reports if there are ops on denylist.

  Print ops if they are on denylist, or print success if no denylisted ops
  found.

  Args:
    meta_graph_def: MetaGraphDef protocol buffer.
    op_denylist: set of ops to scan for.
  """
    ops_in_metagraph = set(meta_graph_lib.ops_used_by_graph_def(meta_graph_def.graph_def))
    denylisted_ops = op_denylist & ops_in_metagraph
    if denylisted_ops:
        print('MetaGraph with tag set %s contains the following denylisted ops:' % meta_graph_def.meta_info_def.tags, denylisted_ops)
    else:
        print('MetaGraph with tag set %s does not contain the default denylisted ops:' % meta_graph_def.meta_info_def.tags, op_denylist)