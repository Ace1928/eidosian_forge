import threading
from absl import logging
from tensorflow.python.distribute.failure_handling.failure_handling_util import detect_platform
from tensorflow.python.distribute.failure_handling.failure_handling_util import PlatformDevice
from tensorflow.python.eager import context
from tensorflow.python.eager import monitoring
from tensorflow.python.framework.errors import AbortedError
from tensorflow.python.framework.errors import CancelledError
from tensorflow.python.framework.errors import InternalError
from tensorflow.python.framework.errors import UnavailableError
from tensorflow.python.util.tf_export import tf_export
@property
def preemption_message(self):
    """Returns the preemption message."""
    return self._preemption_message