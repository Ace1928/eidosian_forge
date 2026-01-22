from typing import Dict, Union
import numpy as np
import tensorflow
from tensorflow.keras.callbacks import TensorBoard
from mlflow.utils.autologging_utils import (
def extract_input_example_from_tf_input_fn(input_fn):
    """
    Extracts sample data from dict (str -> ndarray),
    ``tensorflow.Tensor`` or ``tensorflow.data.Dataset`` type.

    Args:
        input_fn: Tensorflow's input function used for train method

    Returns:
        A slice (of limit ``mlflow.utils.autologging_utils.INPUT_EXAMPLE_SAMPLE_ROWS``)
        of the input of type `np.ndarray`.
        Returns `None` if the return type of ``input_fn`` is unsupported.
    """
    input_training_data = input_fn()
    input_features = None
    if isinstance(input_training_data, tuple):
        features = input_training_data[0]
        if isinstance(features, dict):
            input_features = _extract_sample_numpy_dict(features)
        elif isinstance(features, (np.ndarray, tensorflow.Tensor)):
            input_features = _extract_input_example_from_tensor_or_ndarray(features)
    elif isinstance(input_training_data, tensorflow.data.Dataset):
        input_features = _extract_input_example_from_batched_tf_dataset(input_training_data)
    return input_features