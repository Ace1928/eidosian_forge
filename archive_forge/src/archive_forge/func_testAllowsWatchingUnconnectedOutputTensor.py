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
def testAllowsWatchingUnconnectedOutputTensor(self):
    """Watch an output slot not emitting any edges.

    (Not even control edges from the node.)
    """
    with session.Session() as sess:
        x_init = constant_op.constant([2, 2, 3, 5, 5])
        x = variable_v1.VariableV1(x_init, name='unconnected/x')
        unique_x, _ = array_ops.unique(x, name='unconnected/unique_x')
        y = math_ops.add(unique_x, [0, 1, 2], name='unconnected/y')
        x.initializer.run()
        unique_x_slot_0_recipients = []
        unique_x_slot_1_recipients = []
        for op in sess.graph.get_operations():
            for inp in op.inputs:
                if inp.name == 'unconnected/unique_x:0':
                    unique_x_slot_0_recipients.append(op.name)
                elif inp.name == 'unconnected/unique_x:1':
                    unique_x_slot_1_recipients.append(op.name)
        self.assertEqual(['unconnected/y'], unique_x_slot_0_recipients)
        self.assertEqual([], unique_x_slot_1_recipients)
        y_result, dump = self._debug_run_and_get_dump(sess, y)
        self.assertAllClose([2, 4, 7], y_result)
        unique_x_slot_0_dumps = dump.watch_key_to_data('unconnected/unique_x:0:DebugIdentity')
        self.assertEqual(1, len(unique_x_slot_0_dumps))
        self.assertEqual('unconnected/unique_x', unique_x_slot_0_dumps[0].node_name)
        self.assertEqual(0, unique_x_slot_0_dumps[0].output_slot)
        self.assertAllClose([2, 3, 5], unique_x_slot_0_dumps[0].get_tensor())
        unique_x_slot_1_dumps = dump.watch_key_to_data('unconnected/unique_x:1:DebugIdentity')
        self.assertEqual(1, len(unique_x_slot_1_dumps))
        self.assertEqual('unconnected/unique_x', unique_x_slot_1_dumps[0].node_name)
        self.assertEqual(1, unique_x_slot_1_dumps[0].output_slot)
        self.assertAllClose([0, 0, 1, 2, 2], unique_x_slot_1_dumps[0].get_tensor())