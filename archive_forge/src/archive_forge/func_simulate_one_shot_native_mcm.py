from collections import Counter
from typing import Optional, Sequence
import warnings
from numpy.random import default_rng
import numpy as np
import pennylane as qml
from pennylane.measurements import (
from pennylane.typing import Result
from .initialize_state import create_initial_state
from .apply_operation import apply_operation
from .measure import measure
from .sampling import measure_with_samples
def simulate_one_shot_native_mcm(circuit: qml.tape.QuantumScript, rng=None, prng_key=None, debugger=None, interface=None) -> Result:
    """Simulate a single shot of a single quantum script with native mid-circuit measurements.

    Args:
        circuit (QuantumTape): The single circuit to simulate
        rng (Union[None, int, array_like[int], SeedSequence, BitGenerator, Generator]): A
            seed-like parameter matching that of ``seed`` for ``numpy.random.default_rng``.
            If no value is provided, a default RNG will be used.
        prng_key (Optional[jax.random.PRNGKey]): An optional ``jax.random.PRNGKey``. This is
            the key to the JAX pseudo random number generator. If None, a random key will be
            generated. Only for simulation using JAX.
        debugger (_Debugger): The debugger to use
        interface (str): The machine learning interface to create the initial state with

    Returns:
        tuple(TensorLike): The results of the simulation
        dict: The mid-circuit measurement results of the simulation
    """
    mcm_dict = {}
    state, is_state_batched = get_final_state(circuit, debugger=debugger, interface=interface, mid_measurements=mcm_dict)
    if not np.allclose(np.linalg.norm(state), 1.0):
        return (None, mcm_dict)
    return (measure_final_state(circuit, state, is_state_batched, rng=rng, prng_key=prng_key), mcm_dict)