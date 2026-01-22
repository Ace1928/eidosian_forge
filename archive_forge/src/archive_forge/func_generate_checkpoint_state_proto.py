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
@tf_export(v1=['train.generate_checkpoint_state_proto'])
def generate_checkpoint_state_proto(save_dir, model_checkpoint_path, all_model_checkpoint_paths=None, all_model_checkpoint_timestamps=None, last_preserved_timestamp=None):
    """Generates a checkpoint state proto.

  Args:
    save_dir: Directory where the model was saved.
    model_checkpoint_path: The checkpoint file.
    all_model_checkpoint_paths: List of strings.  Paths to all not-yet-deleted
      checkpoints, sorted from oldest to newest.  If this is a non-empty list,
      the last element must be equal to model_checkpoint_path.  These paths
      are also saved in the CheckpointState proto.
    all_model_checkpoint_timestamps: A list of floats, indicating the number of
      seconds since the Epoch when each checkpoint was generated.
    last_preserved_timestamp: A float, indicating the number of seconds since
      the Epoch when the last preserved checkpoint was written, e.g. due to a
      `keep_checkpoint_every_n_hours` parameter (see
      `tf.train.CheckpointManager` for an implementation).
  Returns:
    CheckpointState proto with model_checkpoint_path and
    all_model_checkpoint_paths updated to either absolute paths or
    relative paths to the current save_dir.

  Raises:
    ValueError: If `all_model_checkpoint_timestamps` was provided but its length
      does not match `all_model_checkpoint_paths`.
  """
    if all_model_checkpoint_paths is None:
        all_model_checkpoint_paths = []
    if not all_model_checkpoint_paths or all_model_checkpoint_paths[-1] != model_checkpoint_path:
        logging.info('%s is not in all_model_checkpoint_paths. Manually adding it.', model_checkpoint_path)
        all_model_checkpoint_paths.append(model_checkpoint_path)
    if all_model_checkpoint_timestamps and len(all_model_checkpoint_timestamps) != len(all_model_checkpoint_paths):
        raise ValueError('Checkpoint timestamps, if provided, must match checkpoint paths (got paths %s and timestamps %s)' % (all_model_checkpoint_paths, all_model_checkpoint_timestamps))
    if not os.path.isabs(save_dir):
        if not os.path.isabs(model_checkpoint_path):
            model_checkpoint_path = os.path.relpath(model_checkpoint_path, save_dir)
        for i, p in enumerate(all_model_checkpoint_paths):
            if not os.path.isabs(p):
                all_model_checkpoint_paths[i] = os.path.relpath(p, save_dir)
    coord_checkpoint_proto = CheckpointState(model_checkpoint_path=model_checkpoint_path, all_model_checkpoint_paths=all_model_checkpoint_paths, all_model_checkpoint_timestamps=all_model_checkpoint_timestamps, last_preserved_timestamp=last_preserved_timestamp)
    return coord_checkpoint_proto