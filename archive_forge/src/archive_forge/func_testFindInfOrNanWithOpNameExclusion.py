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
def testFindInfOrNanWithOpNameExclusion(self):
    with session.Session() as sess:
        u_name = 'testFindInfOrNanWithOpNameExclusion/u'
        v_name = 'testFindInfOrNanWithOpNameExclusion/v'
        w_name = 'testFindInfOrNanWithOpNameExclusion/w'
        x_name = 'testFindInfOrNanWithOpNameExclusion/x'
        y_name = 'testFindInfOrNanWithOpNameExclusion/y'
        z_name = 'testFindInfOrNanWithOpNameExclusion/z'
        u_init = constant_op.constant([2.0, 4.0])
        u = variable_v1.VariableV1(u_init, name=u_name)
        v_init = constant_op.constant([2.0, 1.0])
        v = variable_v1.VariableV1(v_init, name=v_name)
        w = math_ops.subtract(u, v, name=w_name)
        x = math_ops.div(u, w, name=x_name)
        y = math_ops.multiply(w, x, name=y_name)
        z = math_ops.multiply(y, y, name=z_name)
        u.initializer.run()
        v.initializer.run()
        _, dump = self._debug_run_and_get_dump(sess, z, expected_partition_graph_count=self._expected_partition_graph_count)
        bad_data = dump.find(debug_data.has_inf_or_nan, exclude_node_names='.*/x$')
        self.assertLessEqual(2, len(bad_data))
        node_names = [datum.node_name for datum in bad_data]
        self.assertIn(y_name, node_names)
        self.assertIn(z_name, node_names)
        first_bad_datum = dump.find(debug_data.has_inf_or_nan, first_n=1, exclude_node_names='.*/x$')
        self.assertEqual(1, len(first_bad_datum))