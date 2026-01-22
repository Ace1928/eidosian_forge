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
def tpu_ordinal_function(self, host_id):
    """Returns the TPU ordinal fn."""

    def _tpu_ordinal_function(shard_index_in_host):
        """Return the TPU ordinal associated with a shard.

      Required because the enqueue ops are placed on CPU.

      Args:
        shard_index_in_host: the shard index

      Returns:
        The ordinal of the TPU device the shard's infeed should be placed on.
      """
        if self.model_parallelism_enabled:
            replica = self.device_assignment.lookup_replicas(host_id, 0)[shard_index_in_host]
            return self.device_assignment.tpu_ordinal(replica=replica)
        else:
            return shard_index_in_host % self.num_of_cores_per_host
    return _tpu_ordinal_function