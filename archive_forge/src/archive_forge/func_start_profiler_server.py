import datetime
import os
import threading
from tensorflow.python.client import _pywrap_events_writer
from tensorflow.python.eager import context
from tensorflow.python.framework import errors
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.profiler.internal import _pywrap_profiler
from tensorflow.python.util import compat
from tensorflow.python.util.deprecation import deprecated
@deprecated('2020-07-01', 'use `tf.profiler.experimental.server.start`.')
def start_profiler_server(port):
    """Start a profiler grpc server that listens to given port.

  The profiler server will keep the program running even the training finishes.
  Please shutdown the server with CTRL-C. It can be used in both eager mode and
  graph mode. The service defined in
  tensorflow/core/profiler/profiler_service.proto. Please use
  tensorflow/contrib/tpu/profiler/capture_tpu_profile to capture tracable
  file following https://cloud.google.com/tpu/docs/cloud-tpu-tools#capture_trace

  Args:
    port: port profiler server listens to.
  """
    if context.default_execution_mode == context.EAGER_MODE:
        context.ensure_initialized()
    _pywrap_profiler.start_server(port)