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
def testGraphStructureLookupGivesNodesAndAttributes(self):
    u_name, _, _, dump = self._session_run_for_graph_structure_lookup()
    u_read_name = u_name + '/read'
    if test_util.gpu_device_name():
        node_names = dump.nodes(device_name='/job:localhost/replica:0/task:0/device:GPU:0')
    else:
        node_names = dump.nodes()
    self.assertTrue(u_name in node_names)
    self.assertTrue(u_read_name in node_names)
    u_attr = dump.node_attributes(u_name)
    self.assertEqual(dtypes.float32, u_attr['dtype'].type)
    self.assertEqual(1, len(u_attr['shape'].shape.dim))
    self.assertEqual(2, u_attr['shape'].shape.dim[0].size)
    with self.assertRaisesRegexp(ValueError, 'None of the .* device\\(s\\) has a node named '):
        dump.node_attributes('foo')