import copy
import itertools
import json
import warnings
import weakref
import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import input_ops
from tensorflow.python.eager import context
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import keras_export
from tensorflow.tools.docs import doc_controls
from keras.src import backend
from keras.src import callbacks as callbacks_module
from keras.src import optimizers
from keras.src.dtensor import dtensor_api
from keras.src.dtensor import layout_map as layout_map_lib
from keras.src.engine import base_layer
from keras.src.engine import base_layer_utils
from keras.src.engine import compile_utils
from keras.src.engine import data_adapter
from keras.src.engine import input_layer as input_layer_module
from keras.src.engine import steps_per_execution_tuning
from keras.src.engine import training_utils
from keras.src.metrics import base_metric
from keras.src.mixed_precision import loss_scale_optimizer as lso
from keras.src.optimizers import optimizer
from keras.src.optimizers import optimizer_v1
from keras.src.saving import pickle_utils
from keras.src.saving import saving_api
from keras.src.saving import saving_lib
from keras.src.saving import serialization_lib
from keras.src.saving.legacy import serialization
from keras.src.saving.legacy.saved_model import json_utils
from keras.src.saving.legacy.saved_model import model_serialization
from keras.src.utils import generic_utils
from keras.src.utils import io_utils
from keras.src.utils import layer_utils
from keras.src.utils import tf_inspect
from keras.src.utils import tf_utils
from keras.src.utils import traceback_utils
from keras.src.utils import version_utils
from keras.src.utils.mode_keys import ModeKeys
def compute_metrics(self, x, y, y_pred, sample_weight):
    """Update metric states and collect all metrics to be returned.

        Subclasses can optionally override this method to provide custom metric
        updating and collection logic.

        Example:
        ```python
        class MyModel(tf.keras.Sequential):

          def compute_metrics(self, x, y, y_pred, sample_weight):

            # This super call updates `self.compiled_metrics` and returns
            # results for all metrics listed in `self.metrics`.
            metric_results = super(MyModel, self).compute_metrics(
                x, y, y_pred, sample_weight)

            # Note that `self.custom_metric` is not listed in `self.metrics`.
            self.custom_metric.update_state(x, y, y_pred, sample_weight)
            metric_results['custom_metric_name'] = self.custom_metric.result()
            return metric_results
        ```

        Args:
          x: Input data.
          y: Target data.
          y_pred: Predictions returned by the model (output of `model.call(x)`)
          sample_weight: Sample weights for weighting the loss function.

        Returns:
          A `dict` containing values that will be passed to
          `tf.keras.callbacks.CallbackList.on_train_batch_end()`. Typically, the
          values of the metrics listed in `self.metrics` are returned. Example:
          `{'loss': 0.2, 'accuracy': 0.7}`.
        """
    del x
    self.compiled_metrics.update_state(y, y_pred, sample_weight)
    return self.get_metrics_result()