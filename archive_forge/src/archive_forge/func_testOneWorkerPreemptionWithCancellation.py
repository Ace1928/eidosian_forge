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
def testOneWorkerPreemptionWithCancellation(self):

    @def_function.function
    def normal_function():
        x = random_ops.random_uniform((2, 10))
        y = random_ops.random_uniform((10, 2))
        return math_ops.reduce_mean(math_ops.matmul(x, y))

    @def_function.function
    def error_function():
        x = random_ops.random_uniform((2, 10))
        y = random_ops.random_uniform((10, 2))
        check_ops.assert_non_positive_v2(math_ops.reduce_sum(math_ops.matmul(x, y)))
        return x

    @def_function.function
    def long_function():
        x = random_ops.random_uniform((1000, 1000))
        for _ in math_ops.range(10000):
            a = random_ops.random_uniform((1000, 1000))
            b = random_ops.random_uniform((1000, 1000))
            x += math_ops.matmul(a, b)
        return x
    for _ in range(3):
        self.cluster_coord.schedule(normal_function)
    long_function_result = self.cluster_coord.schedule(long_function)
    self.cluster_coord.schedule(error_function)
    time.sleep(1)
    self._restart(2, 'worker')
    with self.assertRaises(errors.InvalidArgumentError):
        self.cluster_coord.join()
    with self.assertRaises(errors.CancelledError):
        long_function_result.fetch()
    for _ in range(3):
        self.cluster_coord.schedule(normal_function)
    self.cluster_coord.join()
    failure_handler = self.cluster_coord._cluster.failure_handler
    failure_handler.stop()
    failure_handler._preemption_handler_thread.join()