import copy
import os
import tensorflow.compat.v2 as tf
import keras.src as keras
from keras.src import backend
from keras.src import losses
from keras.src import optimizers
from keras.src.engine import base_layer_utils
from keras.src.optimizers import optimizer_v1
from keras.src.saving.legacy import serialization
from keras.src.utils import version_utils
from keras.src.utils.io_utils import ask_to_proceed_with_overwrite
from tensorflow.python.platform import tf_logging as logging
def model_call_inputs(model, keep_original_batch_size=False):
    """Inspect model to get its input signature.

    The model's input signature is a list with a single (possibly-nested)
    object. This is due to the Keras-enforced restriction that tensor inputs
    must be passed in as the first argument.

    For example, a model with input {'feature1': <Tensor>, 'feature2': <Tensor>}
    will have input signature:
    [{'feature1': TensorSpec, 'feature2': TensorSpec}]

    Args:
      model: Keras Model object.
      keep_original_batch_size: A boolean indicating whether we want to keep
        using the original batch size or set it to None. Default is `False`,
        which means that the batch dim of the returned input signature will
        always be set to `None`.

    Returns:
      A tuple containing `(args, kwargs)` TensorSpecs of the model call function
      inputs.
      `kwargs` does not contain the `training` argument.
    """
    input_specs = model.save_spec(dynamic_batch=not keep_original_batch_size)
    if input_specs is None:
        return (None, None)
    input_specs = _enforce_names_consistency(input_specs)
    return input_specs