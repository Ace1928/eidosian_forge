from typing import cast, Iterable, List, Sequence, Tuple, TYPE_CHECKING
from cirq import circuits, ops
from cirq.contrib.acquaintance.gates import acquaint, SwapNetworkGate
from cirq.contrib.acquaintance.mutation_utils import expose_acquaintance_gates
Acquaintance strategy for pairs of pairs.

    Implements UpCCGSD ansatz from arXiv:1810.02327.
    