import collections
import warnings
import numpy as np
from tensorflow.python import tf2
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import parameter_server_strategy
from tensorflow.python.distribute import parameter_server_strategy_v2
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.keras import backend
from tensorflow.python.keras import losses
from tensorflow.python.keras import metrics as metrics_module
from tensorflow.python.keras import optimizer_v1
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.distribute import distributed_training_utils
from tensorflow.python.keras.distribute import distributed_training_utils_v1
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.keras.engine import training as training_lib
from tensorflow.python.keras.engine import training_arrays_v1
from tensorflow.python.keras.engine import training_distributed_v1
from tensorflow.python.keras.engine import training_eager_v1
from tensorflow.python.keras.engine import training_generator_v1
from tensorflow.python.keras.engine import training_utils
from tensorflow.python.keras.engine import training_utils_v1
from tensorflow.python.keras.mixed_precision import loss_scale_optimizer
from tensorflow.python.keras.mixed_precision import policy
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.keras.saving import saving_utils
from tensorflow.python.keras.saving.saved_model import model_serialization
from tensorflow.python.keras.utils import data_utils
from tensorflow.python.keras.utils import layer_utils
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.utils.mode_keys import ModeKeys
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.trackable import base as trackable
from tensorflow.python.types import data as data_types
from tensorflow.python.util import nest
def create_training_target(self, target, run_eagerly=False):
    """Create training_target instance and update the self.training_target.

    Note that the input target should just be a tensor or None, and
    corresponding training target will be created based on the output and
    loss_fn.

    Args:
      target: the target tensor for the current output. Could be None.
      run_eagerly: boolean, whether the model is in run_eagerly mode.

    Raises:
      ValueError if the training_target field for the current instance has
      already been populated.
    """
    if self.has_training_target():
        raise ValueError('The training_target field for the _TrainingEndpoint instance has already been populated')
    if run_eagerly:
        self.training_target = _TrainingTarget(None, feedable=True, skip_target_weights=False)
        return
    if self.should_skip_target():
        self.training_target = _TrainingTarget(None)
    else:
        if target is not None and (not backend.is_placeholder(target)):
            feedable = False
            skip_target_weights = True
        else:
            feedable = True
            skip_target_weights = False
        if target is None:
            target_dtype = losses.LABEL_DTYPES_FOR_LOSSES.get(self.loss_fn, backend.dtype(self.output))
            target = backend.placeholder(ndim=len(self.shape), name=self.output_name + '_target', sparse=backend.is_sparse(self.output), dtype=target_dtype)
        self.training_target = _TrainingTarget(target, feedable=feedable, skip_target_weights=skip_target_weights)