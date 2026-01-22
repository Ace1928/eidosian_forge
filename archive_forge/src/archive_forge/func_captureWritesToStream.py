import collections
from collections import OrderedDict
import contextlib
import functools
import gc
import itertools
import math
import os
import random
import re
import tempfile
import threading
import time
import unittest
from absl.testing import parameterized
import numpy as np
from google.protobuf import descriptor_pool
from google.protobuf import text_format
from tensorflow.core.config import flags
from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python import pywrap_sanitizers
from tensorflow.python import tf2
from tensorflow.python.client import device_lib
from tensorflow.python.client import pywrap_tf_session
from tensorflow.python.client import session
from tensorflow.python.compat.compat import forward_compatibility_horizon
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import _test_metrics_util
from tensorflow.python.framework import config
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import gpu_util
from tensorflow.python.framework import importer
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import tfrt_utils
from tensorflow.python.framework import versions
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import control_flow_util_v2
from tensorflow.python.ops import gen_sync_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import script_ops
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.ops import variables
from tensorflow.python.ops.ragged import ragged_ops  # pylint: disable=unused-import
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import ragged_tensor_value
from tensorflow.python.platform import _pywrap_stacktrace_handler
from tensorflow.python.platform import googletest
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import server_lib
from tensorflow.python.util import _pywrap_util_port
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
from tensorflow.python.util import traceback_utils
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.protobuf import compare
from tensorflow.python.util.tf_export import tf_export
@contextlib.contextmanager
def captureWritesToStream(self, stream):
    """A context manager that captures the writes to a given stream.

    This context manager captures all writes to a given stream inside of a
    `CapturedWrites` object. When this context manager is created, it yields
    the `CapturedWrites` object. The captured contents can be accessed  by
    calling `.contents()` on the `CapturedWrites`.

    For this function to work, the stream must have a file descriptor that
    can be modified using `os.dup` and `os.dup2`, and the stream must support
    a `.flush()` method. The default python sys.stdout and sys.stderr are
    examples of this. Note that this does not work in Colab or Jupyter
    notebooks, because those use alternate stdout streams.

    Example:
    ```python
    class MyOperatorTest(test_util.TensorFlowTestCase):
      def testMyOperator(self):
        input = [1.0, 2.0, 3.0, 4.0, 5.0]
        with self.captureWritesToStream(sys.stdout) as captured:
          result = MyOperator(input).eval()
        self.assertStartsWith(captured.contents(), "This was printed.")
    ```

    Args:
      stream: The stream whose writes should be captured. This stream must have
        a file descriptor, support writing via using that file descriptor, and
        must have a `.flush()` method.

    Yields:
      A `CapturedWrites` object that contains all writes to the specified stream
      made during this context.
    """
    stream.flush()
    fd = stream.fileno()
    tmp_file, tmp_file_path = tempfile.mkstemp(dir=self.get_temp_dir())
    orig_fd = os.dup(fd)
    os.dup2(tmp_file, fd)
    try:
        yield CapturedWrites(tmp_file_path)
    finally:
        os.close(tmp_file)
        os.dup2(orig_fd, fd)