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
@staticmethod
def from_devices(session, devices):
    """Construct a heartbeat manager for the given devices."""
    if not devices:
        logging.error('Trying to create heartbeat manager with no devices?')
    logging.info('Creating heartbeat manager for %s', devices)
    request_placeholder = array_ops.placeholder(name='worker_heartbeat_request', dtype=dtypes.string)
    heartbeat_ops = []
    for device in devices:
        with ops.device(device):
            heartbeat_ops.append(tpu_ops.worker_heartbeat(request_placeholder))
    return WorkerHeartbeatManager(session, devices, heartbeat_ops, request_placeholder)