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
def testWorkerExecutionAfterPsFailureRaisesExpectedError(self):
    model = self._create_model_and_run_indefinitely()
    for i in range(self.num_ps):
        self._cluster.kill_task('ps', i)
    while self.cluster_coord._cluster.closure_queue._error is None:
        time.sleep(1)

    @def_function.function
    def trivial_function():
        return model.iterations + 1
    for i in range(self.num_workers):
        try:
            with ops.device('/job:worker/replica:0/task:{}'.format(i)):
                trivial_function()
        except Exception as e:
            if cluster_coordinator._is_ps_failure(e):
                if i < self.num_workers - 1:
                    continue
                return
        raise AssertionError('Executing a function after PS fails, should result in a PS failure.')