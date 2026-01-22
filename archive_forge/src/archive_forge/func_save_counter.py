import atexit
import collections
import copy
import queue
import threading
import time
import weakref
from absl import logging
from tensorflow.python.checkpoint import checkpoint_context
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute.sharded_variable import ShardedVariable
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import executor
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import ops
from tensorflow.python.ops.resource_variable_ops import UninitializedVariable
from tensorflow.python.ops.variables import Variable
from tensorflow.python.saved_model.pywrap_saved_model import metrics
from tensorflow.python.util import object_identity
@property
def save_counter(self):
    """An integer variable numbering the checkpoint events.

    This is maintained by the underlying tf.train.Checkpoing object employed by
    AsyncCheckpoint class. The number starts at 0 and gets incremented for each
    checkpoint event.

    Returns:
      The save counter variable.
    """
    return self.checkpointer().save_counter