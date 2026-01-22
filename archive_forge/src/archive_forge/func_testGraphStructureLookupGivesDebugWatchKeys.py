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
def testGraphStructureLookupGivesDebugWatchKeys(self):
    u_name, v_name, w_name, dump = self._session_run_for_graph_structure_lookup()
    self.assertEqual(['%s:0:DebugIdentity' % u_name], dump.debug_watch_keys(u_name))
    self.assertEqual(['%s:0:DebugIdentity' % v_name], dump.debug_watch_keys(v_name))
    self.assertEqual(['%s:0:DebugIdentity' % w_name], dump.debug_watch_keys(w_name))
    self.assertEqual([], dump.debug_watch_keys('foo'))
    u_data = dump.watch_key_to_data(dump.debug_watch_keys(u_name)[0])
    self.assertEqual(1, len(u_data))
    self.assertEqual(u_name, u_data[0].node_name)
    self.assertEqual(0, u_data[0].output_slot)
    self.assertEqual('DebugIdentity', u_data[0].debug_op)
    self.assertGreaterEqual(u_data[0].timestamp, 0)
    self.assertEqual([], dump.watch_key_to_data('foo'))