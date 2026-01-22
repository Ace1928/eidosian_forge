import warnings
from typing import Any, cast, Iterable, Optional, Sequence, Tuple, TYPE_CHECKING, TypeVar, Union
import numpy as np
from typing_extensions import Protocol
from cirq import linalg, qis
from cirq._doc import doc_private
from cirq.protocols import qid_shape_protocol
from cirq.protocols.decompose_protocol import _try_decompose_into_operations_and_qubits
from cirq.type_workarounds import NotImplementedType
@classmethod
def for_unitary(cls, num_qubits: Optional[int]=None, *, qid_shape: Optional[Tuple[int, ...]]=None) -> 'ApplyUnitaryArgs':
    """A default instance corresponding to an identity matrix.

        Specify exactly one argument.

        Args:
            num_qubits: The number of qubits to make space for in the state.
            qid_shape: A tuple representing the number of quantum levels of each
                qubit the identity matrix applies to. `qid_shape` is (2, 2, 2) for
                a three-qubit identity operation tensor.

        Raises:
            TypeError: If exactly neither `num_qubits` or `qid_shape` is provided or
                both are provided.
        """
    if (num_qubits is None) == (qid_shape is None):
        raise TypeError('Specify exactly one of num_qubits or qid_shape.')
    if num_qubits is not None:
        qid_shape = (2,) * num_qubits
    qid_shape = cast(Tuple[int, ...], qid_shape)
    num_qubits = len(qid_shape)
    state = qis.eye_tensor(qid_shape, dtype=np.complex128)
    return ApplyUnitaryArgs(state, np.empty_like(state), range(num_qubits))