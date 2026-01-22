from typing import Callable, Mapping, Optional, Sequence
import numpy as np
from cirq.circuits import Circuit
from cirq.ops import QubitOrder, QubitOrderOrList
from cirq.sim import final_state_vector
from cirq.value import state_vector_to_probabilities
def linear_xeb_fidelity(circuit: Circuit, bitstrings: Sequence[int], qubit_order: QubitOrderOrList=QubitOrder.DEFAULT, amplitudes: Optional[Mapping[int, complex]]=None) -> float:
    """Estimates XEB fidelity from one circuit using linear estimator."""
    return xeb_fidelity(circuit, bitstrings, qubit_order, amplitudes, estimator=linear_xeb_fidelity_from_probabilities)