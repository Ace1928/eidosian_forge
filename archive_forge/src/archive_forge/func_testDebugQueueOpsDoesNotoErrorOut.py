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
def testDebugQueueOpsDoesNotoErrorOut(self):
    with session.Session() as sess:
        q = data_flow_ops.FIFOQueue(3, 'float', name='fifo_queue')
        q_init = q.enqueue_many(([101.0, 202.0, 303.0],), name='enqueue_many')
        _, dump = self._debug_run_and_get_dump(sess, q_init)
        self.assertTrue(dump.loaded_partition_graphs())
        fifo_queue_tensor = dump.get_tensors('fifo_queue', 0, 'DebugIdentity')[0]
        self.assertIsInstance(fifo_queue_tensor, debug_data.InconvertibleTensorProto)
        self.assertTrue(fifo_queue_tensor.initialized)
        self.assertAllClose([101.0, 202.0, 303.0], dump.get_tensors('enqueue_many/component_0', 0, 'DebugIdentity')[0])