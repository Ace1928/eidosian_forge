import abc
from typing import List, Mapping, Optional, Tuple, TYPE_CHECKING, Sequence
import numpy as np
from cirq import linalg, qis, value
from cirq.sim import simulator, simulation_utils
def measure_state_vector(state_vector: np.ndarray, indices: Sequence[int], *, qid_shape: Optional[Tuple[int, ...]]=None, out: Optional[np.ndarray]=None, seed: 'cirq.RANDOM_STATE_OR_SEED_LIKE'=None) -> Tuple[List[int], np.ndarray]:
    """Performs a measurement of the state in the computational basis.

    This does not modify `state` unless the optional `out` is `state`.

    Args:
        state_vector: The state to be measured. This state vector is assumed to
            be normalized. The state vector must be of size 2 ** integer.  The
            state vector can be of shape (2 ** integer) or (2, 2, ..., 2).
        indices: Which qubits are measured. The `state_vector` is assumed to be
            supplied in big endian order. That is the xth index of v, when
            expressed as a bitstring, has the largest values in the 0th index.
        qid_shape: The qid shape of the `state_vector`.  Specify this argument
            when using qudits.
        out: An optional place to store the result. If `out` is the same as
            the `state_vector` parameter, then `state_vector` will be modified
            inline. If `out` is not None, then the result is put into `out`.
            If `out` is None a new value will be allocated. In all of these
            case out will be the same as the returned ndarray of the method.
            The shape and dtype of `out` will match that of `state_vector` if
            `out` is None, otherwise it will match the shape and dtype of `out`.
        seed: A seed for the pseudorandom number generator.

    Returns:
        A tuple of a list and a numpy array. The list is an array of booleans
        corresponding to the measurement values (ordered by the indices). The
        numpy array is the post measurement state vector. This state vector has
        the same shape and dtype as the input `state_vector`.

    Raises:
        ValueError if the size of state is not a power of 2.
        IndexError if the indices are out of range for the number of qubits
            corresponding to the state.
    """
    shape = qis.validate_qid_shape(state_vector, qid_shape)
    num_qubits = len(shape)
    qis.validate_indices(num_qubits, indices)
    if len(indices) == 0:
        if out is None:
            out = np.copy(state_vector)
        elif out is not state_vector:
            np.copyto(dst=out, src=state_vector)
        return ([], out)
    prng = value.parse_random_state(seed)
    initial_shape = state_vector.shape
    probs = (state_vector * state_vector.conj()).real
    probs = simulation_utils.state_probabilities_by_indices(probs, indices, shape)
    result = prng.choice(len(probs), p=probs)
    meas_shape = tuple((shape[i] for i in indices))
    measurement_bits = value.big_endian_int_to_digits(result, base=meas_shape)
    result_slice = linalg.slice_for_qubits_equal_to(indices, big_endian_qureg_value=result, qid_shape=shape)
    mask = np.ones(shape, dtype=bool)
    mask[result_slice] = False
    if out is None:
        out = np.copy(state_vector)
    elif out is not state_vector:
        np.copyto(dst=out, src=state_vector)
    out.shape = shape
    out[mask] = 0
    out.shape = initial_shape
    out /= np.sqrt(probs[result])
    assert out is not None
    return (measurement_bits, out)