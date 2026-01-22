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
def testWorkerPreemptionErrorTypeWithPythonFunction(self):

    def worker_train_fn():
        x = random_ops.random_uniform((2, 10))
        y = random_ops.random_uniform((10, 2))
        return math_ops.reduce_mean(math_ops.matmul(x, y))

    def run_fn():
        with self.thread_coord.stop_on_exception():
            with ops.device('/job:worker/replica:0/task:0'):
                for _ in range(3):
                    for _ in range(3):
                        worker_train_fn()
                    time.sleep(5)
    run_thread = threading.Thread(target=run_fn)
    run_thread.start()
    time.sleep(1)
    self._restart(2, 'worker')
    try:
        self.thread_coord.join([run_thread])
    except (errors.UnavailableError, errors.AbortedError) as e:
        logging.info('Got exception %r, error message is %s', e, e)
        self.assertIn(_RPC_ERROR_FROM_WORKER, str(e))
        self.assertNotIn(_RPC_ERROR_FROM_PS, str(e))
        self.assertTrue('failed to connect to all addresses' in str(e) or 'Unable to find a context_id' in str(e) or 'Socket closed' in str(e) or ('Connection reset by peer' in str(e)) or ('Transport closed' in str(e)))