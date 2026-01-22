from typing import Dict, Union
import numpy as np
import tensorflow
from tensorflow.keras.callbacks import TensorBoard
from mlflow.utils.autologging_utils import (
def extract_tf_keras_input_example(input_training_data):
    """
    Generates a sample ndarray or dict (str -> ndarray)
    from the input type 'x' for keras ``fit`` or ``fit_generator``

    Args:
        input_training_data: Keras input function used for ``fit`` or ``fit_generator`` methods.

    Returns:
        a slice of type ndarray or
        dict (str -> ndarray) limited to
        ``mlflow.utils.autologging_utils.INPUT_EXAMPLE_SAMPLE_ROWS``.
        Throws ``MlflowException`` exception, if input_training_data is unsupported.
        Returns `None` if the type of input_training_data is unsupported.
    """
    input_data_slice = None
    if isinstance(input_training_data, tensorflow.keras.utils.Sequence):
        input_training_data = input_training_data[:][0]
    if isinstance(input_training_data, (np.ndarray, tensorflow.Tensor)):
        input_data_slice = _extract_input_example_from_tensor_or_ndarray(input_training_data)
    elif isinstance(input_training_data, dict):
        input_data_slice = _extract_sample_numpy_dict(input_training_data)
    elif isinstance(input_training_data, tensorflow.data.Dataset):
        input_data_slice = _extract_input_example_from_batched_tf_dataset(input_training_data)
    return input_data_slice