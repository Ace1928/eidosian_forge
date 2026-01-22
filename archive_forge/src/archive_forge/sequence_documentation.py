import json
import random
import numpy as np
from keras.src.utils import data_utils
from tensorflow.python.util.tf_export import keras_export
Returns a JSON string containing the generator's configuration.

        Args:
            **kwargs: Additional keyword arguments to be passed
                to `json.dumps()`.

        Returns:
            A JSON string containing the tokenizer configuration.
        