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
def tpu_host_placement_function(self):
    """Returns the TPU host place function."""
    master = self.master_job

    def _placement_function(_sentinal=None, replica_id=None, host_id=None):
        """Return the host device given replica_id or host_id."""
        assert _sentinal is None
        if replica_id is not None and host_id is not None:
            raise RuntimeError('replica_id and host_id can have only one non-None value.')
        if master is None:
            return '/replica:0/task:0/device:CPU:0'
        else:
            if replica_id is not None:
                if self.model_parallelism_enabled:
                    return self.device_assignment.host_device(replica=replica_id, job=master)
                else:
                    host_id = replica_id / self.num_of_cores_per_host
            return '/job:%s/task:%d/device:CPU:0' % (master, host_id)
    return _placement_function