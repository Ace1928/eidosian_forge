from typing import Any, cast, Iterable, List, Optional, Sequence, Set, TYPE_CHECKING, Tuple, Union
import itertools
import numpy as np
from cirq import value
from cirq._doc import document
def to_valid_state_vector(state_rep: 'cirq.STATE_VECTOR_LIKE', num_qubits: Optional[int]=None, *, qid_shape: Optional[Sequence[int]]=None, dtype: Optional['DTypeLike']=None, atol: float=1e-07) -> np.ndarray:
    """Verifies the state_rep is valid and converts it to ndarray form.

    This method is used to support passing in an integer representing a
    computational basis state or a full state vector as a representation of
    a pure state.

    Args:
        state_rep: If an int, the state vector returned is the state vector
            corresponding to a computational basis state. If a numpy array
            this is the full state vector. Both of these are validated for
            the given number of qubits, and the state must be properly
            normalized and of the appropriate dtype.
        num_qubits: The number of qubits for the state vector. The state_rep
            must be valid for this number of qubits.
        qid_shape: The expected qid shape of the state vector. Specify this
            argument when using qudits.
        dtype: The numpy dtype of the state vector, will be used when creating
            the state for a computational basis state, or validated against if
            state_rep is a numpy array.
        atol: Numerical tolerance for verifying that the norm of the state
            vector is close to 1.

    Returns:
        A numpy ndarray corresponding to the state vector on the given number of
        qubits.

    Raises:
        ValueError: if `state_vector` is not valid or
            num_qubits != len(qid_shape).
    """
    if isinstance(state_rep, value.ProductState):
        num_qubits = len(state_rep)
    if num_qubits is None and qid_shape is None:
        try:
            qid_shape = infer_qid_shape(state_rep)
        except:
            raise ValueError('Failed to infer the qid shape of the given state. Please specify the qid shape explicitly using either the `num_qubits` or `qid_shape` argument.')
    if qid_shape is None:
        qid_shape = (2,) * cast(int, num_qubits)
    else:
        qid_shape = tuple(qid_shape)
    if num_qubits is None:
        num_qubits = len(qid_shape)
    if num_qubits != len(qid_shape):
        raise ValueError(f'num_qubits != len(qid_shape). num_qubits is <{num_qubits!r}>. qid_shape is <{qid_shape!r}>.')
    if isinstance(state_rep, np.ndarray):
        state_rep = np.copy(state_rep)
    state = quantum_state(state_rep, qid_shape, validate=True, dtype=dtype, atol=atol)
    return cast(np.ndarray, state.state_vector())