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
@deprecation.deprecated(date=None, instructions='Use standard file APIs to delete files with this prefix.')
@tf_export(v1=['train.remove_checkpoint'])
def remove_checkpoint(checkpoint_prefix, checkpoint_format_version=saver_pb2.SaverDef.V2, meta_graph_suffix='meta'):
    """Removes a checkpoint given by `checkpoint_prefix`.

  Args:
    checkpoint_prefix: The prefix of a V1 or V2 checkpoint. Typically the result
      of `Saver.save()` or that of `tf.train.latest_checkpoint()`, regardless of
      sharded/non-sharded or V1/V2.
    checkpoint_format_version: `SaverDef.CheckpointFormatVersion`, defaults to
      `SaverDef.V2`.
    meta_graph_suffix: Suffix for `MetaGraphDef` file. Defaults to 'meta'.
  """
    _delete_file_if_exists(meta_graph_filename(checkpoint_prefix, meta_graph_suffix))
    if checkpoint_format_version == saver_pb2.SaverDef.V2:
        _delete_file_if_exists(checkpoint_prefix + '.index')
        _delete_file_if_exists(checkpoint_prefix + '.data-?????-of-?????')
    else:
        _delete_file_if_exists(checkpoint_prefix)