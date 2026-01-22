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
def patch_tf_keras():
    from tensorflow.python.eager import context
    from wandb.util import parse_version
    if parse_version('2.6.0') <= parse_version(tf.__version__) < parse_version('2.13.0'):
        keras_engine = 'keras.engine'
        try:
            from keras.engine import training
            from keras.engine import training_arrays_v1 as training_arrays
            from keras.engine import training_generator_v1 as training_generator
        except (ImportError, AttributeError):
            wandb.termerror('Unable to patch Tensorflow/Keras')
            logger.exception('exception while trying to patch_tf_keras')
            return
    else:
        keras_engine = 'tensorflow.python.keras.engine'
        from tensorflow.python.keras.engine import training
        try:
            from tensorflow.python.keras.engine import training_arrays_v1 as training_arrays
            from tensorflow.python.keras.engine import training_generator_v1 as training_generator
        except (ImportError, AttributeError):
            try:
                from tensorflow.python.keras.engine import training_arrays, training_generator
            except (ImportError, AttributeError):
                wandb.termerror('Unable to patch Tensorflow/Keras')
                logger.exception('exception while trying to patch_tf_keras')
                return
    training_v2_1 = wandb.util.get_module('tensorflow.python.keras.engine.training_v2')
    training_v2_2 = wandb.util.get_module(f'{keras_engine}.training_v1')
    if training_v2_1:
        old_v2 = training_v2_1.Loop.fit
    elif training_v2_2:
        old_v2 = training.Model.fit
    old_arrays = training_arrays.fit_loop
    old_generator = training_generator.fit_generator

    def set_wandb_attrs(cbk, val_data):
        if isinstance(cbk, WandbCallback):
            if is_generator_like(val_data):
                cbk.generator = val_data
            elif is_dataset(val_data):
                if context.executing_eagerly():
                    cbk.generator = iter(val_data)
                else:
                    wandb.termwarn("Found a validation dataset in graph mode, can't patch Keras.")
            elif isinstance(val_data, tuple) and isinstance(val_data[0], tf.Tensor):

                def gen():
                    while True:
                        yield K.get_session().run(val_data)
                cbk.generator = gen()
            else:
                cbk.validation_data = val_data

    def new_arrays(*args, **kwargs):
        cbks = kwargs.get('callbacks', [])
        val_inputs = kwargs.get('val_inputs')
        val_targets = kwargs.get('val_targets')
        if val_inputs and val_targets:
            for cbk in cbks:
                set_wandb_attrs(cbk, (val_inputs[0], val_targets[0]))
        return old_arrays(*args, **kwargs)

    def new_generator(*args, **kwargs):
        cbks = kwargs.get('callbacks', [])
        val_data = kwargs.get('validation_data')
        if val_data:
            for cbk in cbks:
                set_wandb_attrs(cbk, val_data)
        return old_generator(*args, **kwargs)

    def new_v2(*args, **kwargs):
        cbks = kwargs.get('callbacks', [])
        val_data = kwargs.get('validation_data')
        if val_data:
            for cbk in cbks:
                set_wandb_attrs(cbk, val_data)
        return old_v2(*args, **kwargs)
    training_arrays.orig_fit_loop = old_arrays
    training_arrays.fit_loop = new_arrays
    training_generator.orig_fit_generator = old_generator
    training_generator.fit_generator = new_generator
    wandb.patched['keras'].append([f'{keras_engine}.training_arrays', 'fit_loop'])
    wandb.patched['keras'].append([f'{keras_engine}.training_generator', 'fit_generator'])
    if training_v2_1:
        training_v2_1.Loop.fit = new_v2
        wandb.patched['keras'].append(['tensorflow.python.keras.engine.training_v2.Loop', 'fit'])
    elif training_v2_2:
        training.Model.fit = new_v2
        wandb.patched['keras'].append([f'{keras_engine}.training.Model', 'fit'])