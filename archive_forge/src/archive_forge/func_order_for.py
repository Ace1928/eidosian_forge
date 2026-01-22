from typing import Any, Callable, Iterable, Optional, Tuple, TypeVar, TYPE_CHECKING
from cirq.ops import raw_types
def order_for(self, qubits: Iterable[raw_types.Qid]) -> Tuple[raw_types.Qid, ...]:
    """Returns a qubit tuple ordered corresponding to the basis.

        Args:
            qubits: Qubits that should be included in the basis. (Additional
                qubits may be added into the output by the basis.)

        Returns:
            A tuple of qubits in the same order that their single-qubit
            matrices would be passed into `np.kron` when producing a matrix for
            the entire system.
        """
    return self._explicit_func(qubits)