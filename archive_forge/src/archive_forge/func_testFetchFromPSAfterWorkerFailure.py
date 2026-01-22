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
def testFetchFromPSAfterWorkerFailure(self):
    model = Model(self.cluster_coord)

    def kill_after_delay():
        time.sleep(3)
        logging.info('Killing worker 0')
        self._cluster.kill_task('worker', 0)
        time.sleep(1)
        logging.info('Restarting worker 0')
        self._cluster.start_task('worker', 0)
    kill_thread = threading.Thread(target=kill_after_delay)
    kill_thread.start()
    model.do_infinite_step.assign(True)
    model.schedule_training_functions(1)
    num_reads = 0
    num_reads_after_restart = 0
    read_interval_secs = 0.1
    worker_has_stopped = False
    while num_reads_after_restart <= 5 and num_reads < 200:
        worker_up = context.check_alive('/job:worker/replica:0/task:0')
        if not worker_up:
            worker_has_stopped = True
        if worker_up and worker_has_stopped:
            num_reads_after_restart += 1
        model.join_training_functions()
        start = time.time()
        while time.time() < start + read_interval_secs:
            model.iterations.read_value()
        num_reads += 1
        model.do_infinite_step.assign(True)
        model.schedule_training_functions(1)