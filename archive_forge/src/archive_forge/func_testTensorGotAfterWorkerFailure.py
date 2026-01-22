import gc
import os
import sys
import threading
import time
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import multi_worker_test_base
from tensorflow.python.distribute import parameter_server_strategy_v2
from tensorflow.python.distribute import test_util
from tensorflow.python.distribute.cluster_resolver.cluster_resolver import SimpleClusterResolver
from tensorflow.python.distribute.coordinator import cluster_coordinator
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import coordinator as thread_coordinator
from tensorflow.python.training import server_lib
def testTensorGotAfterWorkerFailure(self):
    with self.strategy.scope():
        v = variables.Variable(initial_value=0, dtype=dtypes.int32)

    @def_function.function
    def worker_fn():
        return (v + 1, v - 1)
    remote_value = self.cluster_coord.schedule(worker_fn)
    fetched = remote_value.get()[0]
    self.assertIsInstance(fetched, tensor.Tensor)
    self.assertEqual(fetched.device, '/job:chief/replica:0/task:0/device:CPU:0')
    self.assertEqual((1, -1), remote_value.get())
    remote_value.get()[0].numpy()
    values = remote_value._values[0]
    self.assertIsInstance(values, tensor.Tensor)
    self.assertRegex(values.device, '/job:worker/replica:0/task:[0-1]/device:CPU:0')
    self.assertEqual((1, -1), remote_value._values)
    remote_value._values[0].numpy()
    for i in range(self.num_workers):
        self._cluster.kill_task('worker', i)
    time.sleep(5)
    remote_value.get()[0].numpy()
    self.assertEqual((1, -1), remote_value.get())
    with self.assertRaises(errors.UnavailableError) as cm:
        remote_value._values[0].numpy()
    self.assertIn('failed to connect to all addresses', cm.exception.message)
    self.assertIn('/job:worker/replica:0/task:', cm.exception.message)