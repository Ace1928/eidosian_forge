import logging
from typing import TYPE_CHECKING, Callable, Dict, Optional, Type, Union
import numpy as np
import tensorflow as tf
from ray.air._internal.tensorflow_utils import convert_ndarray_batch_to_tf_tensor_batch
from ray.train._internal.dl_predictor import DLPredictor
from ray.train.predictor import DataBatchType
from ray.train.tensorflow import TensorflowCheckpoint
from ray.util import log_once
from ray.util.annotations import DeveloperAPI, PublicAPI
@classmethod
def from_checkpoint(cls, checkpoint: TensorflowCheckpoint, model_definition: Optional[Union[Callable[[], tf.keras.Model], Type[tf.keras.Model]]]=None, use_gpu: Optional[bool]=False) -> 'TensorflowPredictor':
    """Instantiate the predictor from a TensorflowCheckpoint.

        Args:
            checkpoint: The checkpoint to load the model and preprocessor from.
            model_definition: A callable that returns a TensorFlow Keras model
                to use. Model weights will be loaded from the checkpoint.
                This is only needed if the `checkpoint` was created from
                `TensorflowCheckpoint.from_model`.
            use_gpu: Whether GPU should be used during prediction.
        """
    if model_definition:
        raise DeprecationWarning('`model_definition` is deprecated. `TensorflowCheckpoint.from_model` now saves the full model definition in .keras format.')
    model = checkpoint.get_model()
    preprocessor = checkpoint.get_preprocessor()
    return cls(model=model, preprocessor=preprocessor, use_gpu=use_gpu)