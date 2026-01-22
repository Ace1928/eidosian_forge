from typing import cast, Iterable, List, Sequence, Tuple, TYPE_CHECKING
from cirq import circuits, ops
from cirq.contrib.acquaintance.gates import acquaint, SwapNetworkGate
from cirq.contrib.acquaintance.mutation_utils import expose_acquaintance_gates
def qubit_pairs_to_qubit_order(qubit_pairs: Sequence[Sequence['cirq.Qid']]) -> List['cirq.Qid']:
    """Takes a sequence of qubit pairs and returns a sequence in which every
    pair is at distance two.

    Specifically, given pairs (1a, 1b), (2a, 2b), etc. returns
    (1a, 2a, 1b, 2b, 3a, 4a, 3b, 4b, ...).
    """
    if set((len(qubit_pair) for qubit_pair in qubit_pairs)) != set((2,)):
        raise ValueError('set(len(qubit_pair) for qubit_pair in qubit_pairs) != set((2,))')
    n_pairs = len(qubit_pairs)
    qubits: List['cirq.Qid'] = []
    for i in range(0, 2 * (n_pairs // 2), 2):
        qubits += [qubit_pairs[i][0], qubit_pairs[i + 1][0], qubit_pairs[i][1], qubit_pairs[i + 1][1]]
    if n_pairs % 2:
        qubits += list(qubit_pairs[-1])
    return qubits