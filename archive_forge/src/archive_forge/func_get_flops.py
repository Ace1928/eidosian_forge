import logging
import operator
import os
import shutil
import sys
from itertools import chain
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K  # noqa: N812
import wandb
from wandb.sdk.integration_utils.data_logging import ValidationDataLogger
from wandb.sdk.lib.deprecate import Deprecated, deprecate
from wandb.util import add_import_hook
def get_flops(self) -> float:
    """Calculate FLOPS [GFLOPs] for a tf.keras.Model or tf.keras.Sequential model in inference mode.

        It uses tf.compat.v1.profiler under the hood.
        """
    if not hasattr(self, 'model'):
        raise wandb.Error('self.model must be set before using this method.')
    if not isinstance(self.model, (tf.keras.models.Sequential, tf.keras.models.Model)):
        raise ValueError('Calculating FLOPS is only supported for `tf.keras.Model` and `tf.keras.Sequential` instances.')
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
    batch_size = 1
    inputs = [tf.TensorSpec([batch_size] + inp.shape[1:], inp.dtype) for inp in self.model.inputs]
    real_model = tf.function(self.model).get_concrete_function(inputs)
    frozen_func, _ = convert_variables_to_constants_v2_as_graph(real_model)
    run_meta = tf.compat.v1.RunMetadata()
    opts = tf.compat.v1.profiler.ProfileOptionBuilder(tf.compat.v1.profiler.ProfileOptionBuilder().float_operation()).with_empty_output().build()
    flops = tf.compat.v1.profiler.profile(graph=frozen_func.graph, run_meta=run_meta, cmd='scope', options=opts)
    return flops.total_float_ops / 1000000000.0 / 2