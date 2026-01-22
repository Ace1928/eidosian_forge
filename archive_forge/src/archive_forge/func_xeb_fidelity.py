from typing import Callable, Mapping, Optional, Sequence
import numpy as np
from cirq.circuits import Circuit
from cirq.ops import QubitOrder, QubitOrderOrList
from cirq.sim import final_state_vector
from cirq.value import state_vector_to_probabilities
def xeb_fidelity(circuit: Circuit, bitstrings: Sequence[int], qubit_order: QubitOrderOrList=QubitOrder.DEFAULT, amplitudes: Optional[Mapping[int, complex]]=None, estimator: Callable[[int, Sequence[float]], float]=linear_xeb_fidelity_from_probabilities) -> float:
    """Estimates XEB fidelity from one circuit using user-supplied estimator.

    Fidelity quantifies the similarity of two quantum states. Here, we estimate
    the fidelity between the theoretically predicted output state of circuit and
    the state produced in its experimental realization. Note that we don't know
    the latter state. Nevertheless, we can estimate the fidelity between the two
    states from the knowledge of the bitstrings observed in the experiment.

    In order to make the estimate more robust one should average the estimates
    over many random circuits. The API supports per-circuit fidelity estimation
    to enable users to examine the properties of estimate distribution over
    many circuits.

    See https://arxiv.org/abs/1608.00263 for more details.

    Args:
        circuit: Random quantum circuit which has been executed on quantum
            processor under test.
        bitstrings: Results of terminal all-qubit measurements performed after
            each circuit execution as integer array where each integer is
            formed from measured qubit values according to `qubit_order` from
            most to least significant qubit, i.e. in the order consistent with
            `cirq.final_state_vector`.
        qubit_order: Qubit order used to construct bitstrings enumerating
            qubits starting with the most significant qubit.
        amplitudes: Optional mapping from bitstring to output amplitude.
            If provided, simulation is skipped. Useful for large circuits
            when an offline simulation had already been performed.
        estimator: Fidelity estimator to use, see above. Defaults to the
            linear XEB fidelity estimator.
    Returns:
        Estimate of fidelity associated with an experimental realization of
        circuit which yielded measurements in bitstrings.
    Raises:
        ValueError: Circuit is inconsistent with qubit order or one of the
            bitstrings is inconsistent with the number of qubits.
    """
    dim = np.prod(circuit.qid_shape()).item()
    if isinstance(bitstrings, tuple):
        bitstrings = list(bitstrings)
    for bitstring in bitstrings:
        if not 0 <= bitstring < dim:
            raise ValueError(f'Bitstring {bitstring} could not have been observed on {len(circuit.qid_shape())} qubits.')
    if amplitudes is None:
        output_state = final_state_vector(circuit, qubit_order=qubit_order)
        output_probabilities = state_vector_to_probabilities(output_state)
        bitstring_probabilities = output_probabilities[bitstrings].tolist()
    else:
        bitstring_probabilities = [abs(amplitudes[bitstring]) ** 2 for bitstring in bitstrings]
    return estimator(dim, bitstring_probabilities)