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
def testDumpUninitializedVariable(self):
    op_namespace = 'testDumpUninitializedVariable'
    with session.Session() as sess:
        u_init_val = np.array([[5.0, 3.0], [-1.0, 0.0]])
        s_init_val = b'str1'
        u_name = '%s/u' % op_namespace
        s_name = '%s/s' % op_namespace
        u_init = constant_op.constant(u_init_val, shape=[2, 2])
        u = variable_v1.VariableV1(u_init, name=u_name)
        s_init = constant_op.constant(s_init_val)
        s = variable_v1.VariableV1(s_init, name=s_name)
        run_options = config_pb2.RunOptions(output_partition_graphs=True)
        debug_urls = self._debug_urls()
        debug_utils.add_debug_tensor_watch(run_options, u_name, 0, debug_urls=debug_urls)
        debug_utils.add_debug_tensor_watch(run_options, s_name, 0, debug_urls=debug_urls)
        run_metadata = config_pb2.RunMetadata()
        sess.run(variables.global_variables_initializer(), options=run_options, run_metadata=run_metadata)
        dump = debug_data.DebugDumpDir(self._dump_root, partition_graphs=run_metadata.partition_graphs)
        self.assertEqual(2, dump.size)
        self.assertEqual(self._expected_partition_graph_count, len(run_metadata.partition_graphs))
        u_vals = dump.get_tensors(u_name, 0, 'DebugIdentity')
        s_vals = dump.get_tensors(s_name, 0, 'DebugIdentity')
        self.assertEqual(1, len(u_vals))
        self.assertIsInstance(u_vals[0], debug_data.InconvertibleTensorProto)
        self.assertFalse(u_vals[0].initialized)
        self.assertEqual(1, len(s_vals))
        self.assertIsInstance(s_vals[0], debug_data.InconvertibleTensorProto)
        self.assertFalse(s_vals[0].initialized)
        self.assertAllClose(u_init_val, sess.run(u))
        self.assertEqual(s_init_val, sess.run(s))