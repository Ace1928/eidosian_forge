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
def with_control_flow_v2(cls):
    """Adds methods that call original methods with WhileV2 and CondV2 enabled.

  Note this enables CondV2 and WhileV2 in new methods after running the test
  class's setup method.

  In addition to this, callers must import the while_v2 module in order to set
  the _while_v2 module in control_flow_ops.

  If a test function has _disable_control_flow_v2 attr set to True (using the
  @disable_control_flow_v2 decorator), the v2 function is not generated for it.

  Example:

  @test_util.with_control_flow_v2
  class ControlFlowTest(test.TestCase):

    def testEnabledForV2(self):
      ...

    @test_util.disable_control_flow_v2("b/xyzabc")
    def testDisabledForV2(self):
      ...

  Generated class:
  class ControlFlowTest(test.TestCase):

    def testEnabledForV2(self):
      ...

    def testEnabledForV2WithControlFlowV2(self):
      // Enable V2 flags.
      testEnabledForV2(self)
      // Restore V2 flags.

    def testDisabledForV2(self):
      ...

  Args:
    cls: class to decorate

  Returns:
    cls with new test methods added
  """
    if control_flow_util.ENABLE_CONTROL_FLOW_V2:
        return cls
    for name, value in cls.__dict__.copy().items():
        if callable(value) and name.startswith(unittest.TestLoader.testMethodPrefix) and (not getattr(value, '_disable_control_flow_v2', False)):
            setattr(cls, name + 'WithControlFlowV2', enable_control_flow_v2(value))
    return cls