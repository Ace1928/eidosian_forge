from typing import Any, Dict, Iterable, Sequence, Tuple, TYPE_CHECKING
from cirq import protocols, value
from cirq.ops import raw_types, swap_gates
Create a `cirq.QubitPermutationGate`.

        Args:
            permutation: A shuffled sequence of integers from 0 to
                len(permutation) - 1. The entry at offset `i` is the result
                of permuting `i`.

        Raises:
            ValueError: If the supplied permutation is not valid (empty, repeated indices, indices
                out of range).
        