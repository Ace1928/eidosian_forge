import abc
from typing import List, Mapping, Optional, Tuple, TYPE_CHECKING, Sequence
import numpy as np
from cirq import linalg, qis, value
from cirq.sim import simulator, simulation_utils
def sample_state_vector(state_vector: np.ndarray, indices: Sequence[int], *, qid_shape: Optional[Tuple[int, ...]]=None, repetitions: int=1, seed: 'cirq.RANDOM_STATE_OR_SEED_LIKE'=None) -> np.ndarray:
    """Samples repeatedly from measurements in the computational basis.

    Note that this does not modify the passed in state.

    Args:
        state_vector: The multi-qubit state vector to be sampled. This is an
            array of 2 to the power of the number of qubit complex numbers, and
            so state must be of size ``2**integer``.  The `state_vector` can be
            a vector of size ``2**integer`` or a tensor of shape
            ``(2, 2, ..., 2)``.
        indices: Which qubits are measured. The `state_vector` is assumed to be
            supplied in big endian order. That is the xth index of v, when
            expressed as a bitstring, has its largest values in the 0th index.
        qid_shape: The qid shape of the `state_vector`.  Specify this argument
            when using qudits.
        repetitions: The number of times to sample.
        seed: A seed for the pseudorandom number generator.

    Returns:
        Measurement results with True corresponding to the ``|1⟩`` state.
        The outer list is for repetitions, and the inner corresponds to
        measurements ordered by the supplied qubits. These lists
        are wrapped as a numpy ndarray.

    Raises:
        ValueError: ``repetitions`` is less than one or size of `state_vector`
            is not a power of 2.
        IndexError: An index from ``indices`` is out of range, given the number
            of qubits corresponding to the state.
    """
    if repetitions < 0:
        raise ValueError(f'Number of repetitions cannot be negative. Was {repetitions}')
    shape = qis.validate_qid_shape(state_vector, qid_shape)
    num_qubits = len(shape)
    qis.validate_indices(num_qubits, indices)
    if repetitions == 0 or len(indices) == 0:
        return np.zeros(shape=(repetitions, len(indices)), dtype=np.uint8)
    prng = value.parse_random_state(seed)
    probs = (state_vector * state_vector.conj()).real
    probs = simulation_utils.state_probabilities_by_indices(probs, indices, shape)
    result = prng.choice(len(probs), size=repetitions, p=probs)
    meas_shape = tuple((shape[i] for i in indices))
    return np.array([value.big_endian_int_to_digits(result[i], base=meas_shape) for i in range(len(result))], dtype=np.uint8)