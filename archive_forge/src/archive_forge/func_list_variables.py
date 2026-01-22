from collections import abc
import os
import time
from tensorflow.python.checkpoint import checkpoint_management
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.framework import ops
from tensorflow.python.ops import io_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import variables
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import py_checkpoint_reader
from tensorflow.python.training.saving import saveable_object_util
from tensorflow.python.util.tf_export import tf_export
@tf_export('train.list_variables')
def list_variables(ckpt_dir_or_file):
    """Lists the checkpoint keys and shapes of variables in a checkpoint.

  Checkpoint keys are paths in a checkpoint graph.

  Example usage:

  ```python
  import tensorflow as tf
  import os
  ckpt_directory = "/tmp/training_checkpoints/ckpt"
  ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
  manager = tf.train.CheckpointManager(ckpt, ckpt_directory, max_to_keep=3)
  train_and_checkpoint(model, manager)
  tf.train.list_variables(manager.latest_checkpoint)
  ```

  Args:
    ckpt_dir_or_file: Directory with checkpoints file or path to checkpoint.

  Returns:
    List of tuples `(key, shape)`.
  """
    reader = load_checkpoint(ckpt_dir_or_file)
    variable_map = reader.get_variable_to_shape_map()
    names = sorted(variable_map.keys())
    result = []
    for name in names:
        result.append((name, variable_map[name]))
    return result