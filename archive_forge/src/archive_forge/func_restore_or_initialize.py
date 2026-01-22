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
def restore_or_initialize(self):
    """Restore items in `checkpoint` from the latest checkpoint file.

    This method will first try to restore from the most recent checkpoint in
    `directory`. If no checkpoints exist in `directory`, and `init_fn` is
    specified, this method will call `init_fn` to do customized
    initialization. This can be used to support initialization from pretrained
    models.

    Note that unlike `tf.train.Checkpoint.restore()`, this method doesn't return
    a load status object that users can run assertions on
    (e.g. assert_consumed()). Thus to run assertions, users should directly use
    `tf.train.Checkpoint.restore()` method.

    Returns:
      The restored checkpoint path if the lastest checkpoint is found and
      restored. Otherwise None.
    """
    if self._latest_checkpoint is not None:
        self._checkpoint.restore(self._latest_checkpoint)
        if self._checkpoint_interval is not None:
            self._last_checkpoint_step = _evaluate(self._step_counter)
        return self._latest_checkpoint
    if self._init_fn is not None:
        self._init_fn()
        logging.info('Customized initialization is done through the passed `init_fn`.')
    return None