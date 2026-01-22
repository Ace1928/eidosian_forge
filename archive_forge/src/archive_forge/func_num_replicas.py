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
def num_replicas(self):
    """Compute the total number of replicas."""
    num_cores_in_system = self.num_cores
    if self.model_parallelism_enabled:
        num_cores_per_replica = self._config.tpu_config.num_cores_per_replica
        if num_cores_per_replica > num_cores_in_system:
            raise ValueError('The num of cores required by the model parallelism, specified by TPUConfig.num_cores_per_replica, is larger than the total num of TPU cores in the system. num_cores_per_replica: {}, num cores in the system: {}'.format(num_cores_per_replica, num_cores_in_system))
        if num_cores_in_system % num_cores_per_replica != 0:
            raise RuntimeError('The num of cores in the system ({}) is not divisible by the num of cores ({}) required by the model parallelism, specified by TPUConfig.num_cores_per_replica. This should never happen!'.format(num_cores_in_system, num_cores_per_replica))
        return num_cores_in_system // num_cores_per_replica
    else:
        return num_cores_in_system