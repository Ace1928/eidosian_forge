import math
import numpy as np
import tree
from keras.src.api_export import keras_export
Returns the list of input tensors necessary to compute `tensor`.

    Output will always be a list of tensors
    (potentially with 1 element).

    Args:
        tensor: The tensor to start from.

    Returns:
        List of input tensors.
    