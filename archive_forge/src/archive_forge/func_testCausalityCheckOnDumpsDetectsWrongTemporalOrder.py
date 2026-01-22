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
def testCausalityCheckOnDumpsDetectsWrongTemporalOrder(self):
    with session.Session(config=no_rewrite_session_config()) as sess:
        u_name = 'testDumpCausalityCheck/u'
        v_name = 'testDumpCausalityCheck/v'
        w_name = 'testDumpCausalityCheck/w'
        u_init = constant_op.constant([2.0, 4.0])
        u = variable_v1.VariableV1(u_init, name=u_name)
        v = math_ops.add(u, u, name=v_name)
        w = math_ops.add(v, v, name=w_name)
        u.initializer.run()
        run_options = config_pb2.RunOptions(output_partition_graphs=True)
        debug_utils.watch_graph(run_options, sess.graph, debug_ops=['DebugIdentity'], debug_urls=self._debug_urls())
        run_metadata = config_pb2.RunMetadata()
        sess.run(w, options=run_options, run_metadata=run_metadata)
        self.assertEqual(self._expected_partition_graph_count, len(run_metadata.partition_graphs))
        debug_data.DebugDumpDir(self._dump_root)
        dump = debug_data.DebugDumpDir(self._dump_root, partition_graphs=run_metadata.partition_graphs)
        self.assertEqual(1, len(dump.get_tensor_file_paths(v_name, 0, 'DebugIdentity')))
        v_file_path = dump.get_tensor_file_paths(v_name, 0, 'DebugIdentity')[0]
        self.assertEqual(1, len(dump.get_tensor_file_paths(w_name, 0, 'DebugIdentity')))
        w_file_path = dump.get_tensor_file_paths(w_name, 0, 'DebugIdentity')[0]
        v_timestamp = int(v_file_path[v_file_path.rindex('_') + 1:])
        w_timestamp = int(w_file_path[w_file_path.rindex('_') + 1:])
        v_file_path_1 = v_file_path[:v_file_path.rindex('_')] + '_%d' % w_timestamp
        w_file_path_1 = w_file_path[:w_file_path.rindex('_')] + '_%d' % (v_timestamp - 1)
        os.rename(v_file_path, v_file_path_1)
        os.rename(w_file_path, w_file_path_1)
        with self.assertRaisesRegexp(ValueError, 'Causality violated'):
            dump = debug_data.DebugDumpDir(self._dump_root, partition_graphs=run_metadata.partition_graphs)
        dump = debug_data.DebugDumpDir(self._dump_root, partition_graphs=run_metadata.partition_graphs, validate=False)
        v_file_path_2 = v_file_path[:v_file_path.rindex('_')] + '_%d' % w_timestamp
        w_file_path_2 = w_file_path[:w_file_path.rindex('_')] + '_%d' % w_timestamp
        os.rename(v_file_path_1, v_file_path_2)
        os.rename(w_file_path_1, w_file_path_2)
        debug_data.DebugDumpDir(self._dump_root, partition_graphs=run_metadata.partition_graphs)