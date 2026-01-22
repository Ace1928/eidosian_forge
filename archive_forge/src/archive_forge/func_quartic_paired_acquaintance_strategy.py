from typing import cast, Iterable, List, Sequence, Tuple, TYPE_CHECKING
from cirq import circuits, ops
from cirq.contrib.acquaintance.gates import acquaint, SwapNetworkGate
from cirq.contrib.acquaintance.mutation_utils import expose_acquaintance_gates
def quartic_paired_acquaintance_strategy(qubit_pairs: Iterable[Tuple['cirq.Qid', ops.Qid]]) -> Tuple['cirq.Circuit', Sequence['cirq.Qid']]:
    """Acquaintance strategy for pairs of pairs.

    Implements UpCCGSD ansatz from arXiv:1810.02327.
    """
    qubit_pairs = tuple((cast(Tuple['cirq.Qid', ops.Qid], tuple(qubit_pair)) for qubit_pair in qubit_pairs))
    qubits = qubit_pairs_to_qubit_order(qubit_pairs)
    n_qubits = len(qubits)
    swap_network = SwapNetworkGate((1,) * n_qubits, 2)(*qubits)
    strategy = circuits.Circuit(swap_network)
    expose_acquaintance_gates(strategy)
    for i in reversed(range(0, n_qubits, 2)):
        moment = circuits.Moment([acquaint(*qubits[j:j + 4]) for j in range(i % 4, n_qubits - 3, 4)])
        strategy.insert(2 * i, moment)
    return (strategy, qubits)