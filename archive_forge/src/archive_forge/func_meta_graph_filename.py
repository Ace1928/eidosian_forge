import collections
import os.path
import re
import time
from google.protobuf import text_format
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.eager import context
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import training_util
from tensorflow.python.training.checkpoint_state_pb2 import CheckpointState
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
def meta_graph_filename(checkpoint_filename, meta_graph_suffix='meta'):
    """Returns the meta graph filename.

  Args:
    checkpoint_filename: Name of the checkpoint file.
    meta_graph_suffix: Suffix for `MetaGraphDef` file. Defaults to 'meta'.

  Returns:
    MetaGraph file name.
  """
    basename = re.sub('-[\\d\\?]+-of-\\d+$', '', checkpoint_filename)
    suffixed_filename = '.'.join([basename, meta_graph_suffix])
    return suffixed_filename