from typing import Sequence, Callable
from functools import partial
import warnings
import numpy as np
import pennylane as qml
from pennylane.pulse import ParametrizedEvolution, HardwareHamiltonian
from pennylane import transform
from .parameter_shift import _make_zero_rep
from .general_shift_rules import eigvals_to_frequencies, generate_shift_rule
from .gradient_transform import (
def raise_pulse_diff_on_qnode(transform_name):
    """Raises an error as the gradient transform with the provided name does
    not support direct application to QNodes.
    """
    msg = (f'Applying the {transform_name} gradient transform to a QNode directly is currently not supported. Please use differentiation via a JAX entry point (jax.grad, jax.jacobian, ...) instead.', UserWarning)
    raise NotImplementedError(msg)