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
def is_running_on_cpu(self, is_export_mode=False):
    """Determines whether the input_fn and model_fn should be invoked on CPU.

    This API also validates user provided configuration, such as batch size,
    according the lazy initialized TPU system metadata.

    Args:
      is_export_mode: Indicates whether the current mode is for exporting the
        model, when mode == PREDICT. Only with this bool, we could tell whether
        user is calling the Estimator.predict or Estimator.export_savedmodel,
        which are running on TPU and CPU respectively. Parent class Estimator
        does not distinguish these two.

    Returns:
      bool, whether current input_fn or model_fn should be running on CPU.

    Raises:
      ValueError: any configuration is invalid.
    """
    is_running_on_cpu = self._is_running_on_cpu(is_export_mode)
    if not is_running_on_cpu:
        self._validate_tpu_configuration()
    return is_running_on_cpu