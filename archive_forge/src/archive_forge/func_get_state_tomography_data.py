from typing import List, Optional, Sequence, Tuple, TYPE_CHECKING
import numpy as np
import sympy
from cirq import circuits, ops, protocols, study
from cirq.experiments.qubit_characterizations import TomographyResult
def get_state_tomography_data(sampler: 'cirq.Sampler', qubits: Sequence['cirq.Qid'], circuit: 'cirq.Circuit', rot_circuit: 'cirq.Circuit', rot_sweep: 'cirq.Sweep', repetitions: int=1000) -> np.ndarray:
    """Gets the data for each rotation string added to the circuit.

    For each sequence in prerotation_sequences gets the probability of all
    2**n bit strings.  Resulting matrix will have dimensions
    (len(rot_sweep)**n, 2**n).
    This is a default way to get data that can be replaced by the user if they
    have a more advanced protocol in mind.

    Args:
        sampler: Sampler to collect the data from.
        qubits: Qubits to do the tomography on.
        circuit: Circuit to do the tomography on.
        rot_circuit: Circuit with parameterized rotation gates to do before the
            final measurements.
        rot_sweep: The list of rotations on the qubits to perform before
            measurement.
        repetitions: Number of times to sample each rotation sequence.

    Returns:
        2D array of probabilities, where first index is which pre-rotation was
        applied and second index is the qubit state.
    """
    results = sampler.run_sweep(circuit + rot_circuit + [ops.measure(*qubits, key='z')], params=rot_sweep, repetitions=repetitions)
    all_probs = []
    for result in results:
        hist = result.histogram(key='z')
        probs = [hist[i] for i in range(2 ** len(qubits))]
        all_probs.append(np.array(probs) / repetitions)
    return np.array(all_probs)