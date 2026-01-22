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
def testDebuggingDuringOpError(self):
    """Test the debug tensor dumping when error occurs in graph runtime."""
    with session.Session() as sess:
        ph = array_ops.placeholder(dtypes.float32, name='mismatch/ph')
        x = array_ops.transpose(ph, name='mismatch/x')
        m = constant_op.constant(np.array([[1.0, 2.0]], dtype=np.float32), name='mismatch/m')
        y = math_ops.matmul(m, x, name='mismatch/y')
        run_options = config_pb2.RunOptions(output_partition_graphs=True)
        debug_utils.watch_graph(run_options, sess.graph, debug_ops=['DebugIdentity'], debug_urls=self._debug_urls())
        with self.assertRaises(errors.OpError):
            sess.run(y, options=run_options, feed_dict={ph: np.array([[-3.0], [0.0]])})
        dump = debug_data.DebugDumpDir(self._dump_root)
        self.assertGreaterEqual(dump.core_metadata.session_run_index, 0)
        self.assertGreaterEqual(dump.core_metadata.executor_step_index, 0)
        self.assertEqual([ph.name], dump.core_metadata.input_names)
        self.assertEqual([y.name], dump.core_metadata.output_names)
        self.assertEqual([], dump.core_metadata.target_nodes)
        self.assertTrue(dump.loaded_partition_graphs())
        m_dumps = dump.watch_key_to_data('mismatch/m:0:DebugIdentity')
        self.assertEqual(1, len(m_dumps))
        self.assertAllClose(np.array([[1.0, 2.0]]), m_dumps[0].get_tensor())
        x_dumps = dump.watch_key_to_data('mismatch/x:0:DebugIdentity')
        self.assertEqual(1, len(x_dumps))
        self.assertAllClose(np.array([[-3.0, 0.0]]), x_dumps[0].get_tensor())