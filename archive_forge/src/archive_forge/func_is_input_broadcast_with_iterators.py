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
def is_input_broadcast_with_iterators(self):
    """Return true if input_fn should be run in the full_replicae config."""
    return self._config.tpu_config.per_host_input_for_training is tpu_config.InputPipelineConfig.BROADCAST or self.is_input_slice_broadcast_to_all_cores()