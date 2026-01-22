import threading
import time
from google.protobuf import text_format
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.util import event_pb2
from tensorflow.python.client import session as session_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.tpu.ops import tpu_ops
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import training_util
def start_worker_watchdog(session, devices=None, ping_interval=60, shutdown_timeout=3600):
    """Start global worker watchdog to shutdown workers on coordinator exit."""
    global _WATCHDOG
    if _WATCHDOG is None:
        ping_interval = min(shutdown_timeout / 10.0, ping_interval)
        _WATCHDOG = WatchdogManager(session, devices, ping_interval, shutdown_timeout)
        _WATCHDOG.configure_and_run()