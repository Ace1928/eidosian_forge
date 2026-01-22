from collections import OrderedDict
from gymnasium.spaces import Discrete, MultiDiscrete
import numpy as np
import tree  # pip install dm_tree
from types import MappingProxyType
from typing import List, Optional
from ray.rllib.utils.annotations import PublicAPI
from ray.rllib.utils.deprecation import (
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.typing import SpaceStruct, TensorType, TensorStructType, Union
@PublicAPI
def lstm(x, weights: np.ndarray, biases: Optional[np.ndarray]=None, initial_internal_states: Optional[np.ndarray]=None, time_major: bool=False, forget_bias: float=1.0):
    """Calculates LSTM layer output given weights/biases, states, and input.

    Args:
        x: The inputs to the LSTM layer including time-rank
            (0th if time-major, else 1st) and the batch-rank
            (1st if time-major, else 0th).
        weights: The weights matrix.
        biases: The biases vector. All 0s if None.
        initial_internal_states: The initial internal
            states to pass into the layer. All 0s if None.
        time_major: Whether to use time-major or not. Default: False.
        forget_bias: Gets added to first sigmoid (forget gate) output.
            Default: 1.0.

    Returns:
        Tuple consisting of 1) The LSTM layer's output and
        2) Tuple: Last (c-state, h-state).
    """
    sequence_length = x.shape[0 if time_major else 1]
    batch_size = x.shape[1 if time_major else 0]
    units = weights.shape[1] // 4
    if initial_internal_states is None:
        c_states = np.zeros(shape=(batch_size, units))
        h_states = np.zeros(shape=(batch_size, units))
    else:
        c_states = initial_internal_states[0]
        h_states = initial_internal_states[1]
    if time_major:
        unrolled_outputs = np.zeros(shape=(sequence_length, batch_size, units))
    else:
        unrolled_outputs = np.zeros(shape=(batch_size, sequence_length, units))
    for t in range(sequence_length):
        input_matrix = x[t, :, :] if time_major else x[:, t, :]
        input_matrix = np.concatenate((input_matrix, h_states), axis=1)
        input_matmul_matrix = np.matmul(input_matrix, weights) + biases
        sigmoid_1 = sigmoid(input_matmul_matrix[:, units * 2:units * 3] + forget_bias)
        c_states = np.multiply(c_states, sigmoid_1)
        sigmoid_2 = sigmoid(input_matmul_matrix[:, 0:units])
        tanh_3 = np.tanh(input_matmul_matrix[:, units:units * 2])
        c_states = np.add(c_states, np.multiply(sigmoid_2, tanh_3))
        sigmoid_4 = sigmoid(input_matmul_matrix[:, units * 3:units * 4])
        h_states = np.multiply(sigmoid_4, np.tanh(c_states))
        if time_major:
            unrolled_outputs[t, :, :] = h_states
        else:
            unrolled_outputs[:, t, :] = h_states
    return (unrolled_outputs, (c_states, h_states))