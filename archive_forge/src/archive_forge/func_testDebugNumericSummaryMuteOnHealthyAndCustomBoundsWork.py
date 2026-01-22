import collections
import functools
import glob
import os
import tempfile
import threading
import numpy as np
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.core.util import event_pb2
from tensorflow.python.client import session
from tensorflow.python.debug.lib import debug_data
from tensorflow.python.debug.lib import debug_graphs
from tensorflow.python.debug.lib import debug_utils
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond as tf_cond
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables
from tensorflow.python.ops import while_loop
import tensorflow.python.ops.tensor_array_grad  # pylint: disable=unused-import
from tensorflow.python.platform import googletest
from tensorflow.python.platform import test
from tensorflow.python.training import gradient_descent
def testDebugNumericSummaryMuteOnHealthyAndCustomBoundsWork(self):
    with session.Session() as sess:
        a = variable_v1.VariableV1([10.0, 10.0], name='a')
        b = variable_v1.VariableV1([10.0, 2.0], name='b')
        x = math_ops.add(a, b, name='x')
        y = math_ops.divide(x, b, name='y')
        sess.run(variables.global_variables_initializer())
        _, dump = self._debug_run_and_get_dump(sess, y, debug_ops=['DebugNumericSummary(mute_if_healthy=true; upper_bound=11.0)'], validate=False)
        self.assertEqual(1, dump.size)
        self.assertAllClose([[1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 12.0, 20.0, 16.0, 16.0, 1.0, 1.0, 2.0]], dump.get_tensors('x', 0, 'DebugNumericSummary'))