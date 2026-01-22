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
def testDebugNumericSummaryFailureIsToleratedWhenOrdered(self):
    with session.Session() as sess:
        a = variable_v1.VariableV1('1', name='a')
        b = variable_v1.VariableV1('3', name='b')
        c = variable_v1.VariableV1('2', name='c')
        d = math_ops.add(a, b, name='d')
        e = math_ops.add(d, c, name='e')
        n = parsing_ops.string_to_number(e, name='n')
        m = math_ops.add(n, n, name='m')
        sess.run(variables.global_variables_initializer())
        run_metadata = config_pb2.RunMetadata()
        run_options = config_pb2.RunOptions(output_partition_graphs=True)
        debug_utils.watch_graph(run_options, sess.graph, debug_ops=['DebugNumericSummary'], debug_urls=self._debug_urls())
        with self.assertRaises(errors.FailedPreconditionError):
            sess.run(m, options=run_options, run_metadata=run_metadata)
        m_result, dump = self._debug_run_and_get_dump(sess, m, debug_ops=['DebugNumericSummary'], tolerate_debug_op_creation_failures=True)
        self.assertEqual(264, m_result)
        self.assertIn('n:0:DebugNumericSummary', dump.debug_watch_keys('n'))
        self.assertIn('m:0:DebugNumericSummary', dump.debug_watch_keys('m'))