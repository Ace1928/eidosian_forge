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
def testTwoWorkersPreempted(self):
    if self.num_workers < 2:
        self.skipTest('Worker number is less than 2.')
    model = self._create_model_and_run_indefinitely()
    self.assertFalse(self.cluster_coord.done())
    self._cluster.kill_task('worker', 0)
    self._cluster.kill_task('worker', 1)
    time.sleep(2)
    self.assertFalse(context.check_alive('/job:worker/replica:0/task:0'))
    self.assertFalse(context.check_alive('/job:worker/replica:0/task:1'))
    self._cluster.start_task('worker', 0)
    self._cluster.start_task('worker', 1)
    time.sleep(2)
    self.assertTrue(context.check_alive('/job:worker/replica:0/task:0'))
    self.assertTrue(context.check_alive('/job:worker/replica:0/task:1'))
    model.join_training_functions()
    self.assertGreaterEqual(model.iterations.numpy(), 10)