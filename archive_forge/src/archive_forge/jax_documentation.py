import logging
from typing import Tuple, Callable
import dataclasses
import jax
import jax.numpy as jnp
import pennylane as qml
from pennylane.transforms import convert_to_numpy_parameters
from pennylane.typing import ResultBatch
Execute a batch of tapes with JAX parameters using JVP derivatives.

    Args:
        tapes (Sequence[.QuantumTape]): batch of tapes to execute
        execute_fn (Callable[[Sequence[.QuantumTape]], ResultBatch]): a function that turns a batch of circuits into results
        jpc (JacobianProductCalculator): a class that can compute the Jacobian vector product (JVP)
            for the input tapes.

    Returns:
        TensorLike: A nested tuple of tape results. Each element in
        the returned tuple corresponds in order to the provided tapes.

    