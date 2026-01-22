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
@tf_export('train.load_checkpoint')
def load_checkpoint(ckpt_dir_or_file):
    """Returns `CheckpointReader` for checkpoint found in `ckpt_dir_or_file`.

  If `ckpt_dir_or_file` resolves to a directory with multiple checkpoints,
  reader for the latest checkpoint is returned.

  Example usage:

  ```python
  import tensorflow as tf
  a = tf.Variable(1.0)
  b = tf.Variable(2.0)
  ckpt = tf.train.Checkpoint(var_list={'a': a, 'b': b})
  ckpt_path = ckpt.save('tmp-ckpt')
  reader= tf.train.load_checkpoint(ckpt_path)
  print(reader.get_tensor('var_list/a/.ATTRIBUTES/VARIABLE_VALUE'))  # 1.0
  ```

  Args:
    ckpt_dir_or_file: Directory with checkpoints file or path to checkpoint
      file.

  Returns:
    `CheckpointReader` object.

  Raises:
    ValueError: If `ckpt_dir_or_file` resolves to a directory with no
      checkpoints.
  """
    filename = _get_checkpoint_filename(ckpt_dir_or_file)
    if filename is None:
        raise ValueError("Couldn't find 'checkpoint' file or checkpoints in given directory %s" % ckpt_dir_or_file)
    return py_checkpoint_reader.NewCheckpointReader(filename)