from typing import Any, Callable, Iterable, Optional, Tuple, TypeVar, TYPE_CHECKING
from cirq.ops import raw_types
@staticmethod
def sorted_by(key: Callable[[raw_types.Qid], Any]) -> 'QubitOrder':
    """A basis that orders qubits ascending based on a key function.

        Args:
            key: A function that takes a qubit and returns a key value. The
                basis will be ordered ascending according to these key values.

        Returns:
            A basis that orders qubits ascending based on a key function.
        """
    return QubitOrder(lambda qubits: tuple(sorted(qubits, key=key)))