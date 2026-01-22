import logging
from typing import Tuple, Callable
import dataclasses
import jax
import jax.numpy as jnp
import pennylane as qml
from pennylane.transforms import convert_to_numpy_parameters
from pennylane.typing import ResultBatch
def set_parameters_on_copy_and_unwrap(tapes, params, unwrap=True):
    """Copy a set of tapes with operations and set parameters"""
    return tuple((_set_copy_and_unwrap_tape(t, a, unwrap=unwrap) for t, a in zip(tapes, params)))