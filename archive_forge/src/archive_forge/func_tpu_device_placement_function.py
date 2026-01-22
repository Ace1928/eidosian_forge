from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from contextlib import contextmanager
import copy
import tensorflow as tf
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.tpu import device_assignment as tpu_device_assignment
from tensorflow.python.tpu import tpu_system_metadata as tpu_system_metadata_lib
from tensorflow_estimator.python.estimator import model_fn as model_fn_lib
from tensorflow_estimator.python.estimator.tpu import _tpu_estimator_embedding
from tensorflow_estimator.python.estimator.tpu import tpu_config
@property
def tpu_device_placement_function(self):
    """Returns a TPU device placement Fn."""
    master = self.master_job
    job_device = '' if master is None else '/job:%s' % master

    def _placement_function(i):
        if self.model_parallelism_enabled:
            return self.device_assignment.tpu_device(replica=i, job=master)
        else:
            num_of_cores_per_host = self.num_of_cores_per_host
            host_id = i / num_of_cores_per_host
            ordinal_id = i % num_of_cores_per_host
            return '%s/task:%d/device:TPU:%d' % (job_device, host_id, ordinal_id)
    return _placement_function