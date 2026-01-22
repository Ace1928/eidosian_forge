from typing import Sequence, TYPE_CHECKING
from cirq import circuits, ops
from cirq.contrib.acquaintance.gates import acquaint
from cirq.contrib.acquaintance.mutation_utils import (
def complete_acquaintance_strategy(qubit_order: Sequence['cirq.Qid'], acquaintance_size: int=0, swap_gate: 'cirq.Gate'=ops.SWAP) -> 'cirq.Circuit':
    """Returns an acquaintance strategy with can handle the given number of qubits.

    Args:
        qubit_order: The qubits on which the strategy should be defined.
        acquaintance_size: The maximum number of qubits to be acted on by
        an operation.
        swap_gate: The gate used to swap logical indices.

    Returns:
        A circuit capable of implementing any set of k-local operations.

    Raises:
        ValueError: If `acquaintance_size` is negative.
    """
    if acquaintance_size < 0:
        raise ValueError('acquaintance_size must be non-negative.')
    if acquaintance_size == 0:
        return circuits.Circuit()
    if acquaintance_size > len(qubit_order):
        return circuits.Circuit()
    if acquaintance_size == len(qubit_order):
        return circuits.Circuit(acquaint(*qubit_order))
    strategy = circuits.Circuit((acquaint(q) for q in qubit_order))
    for size_to_acquaint in range(2, acquaintance_size + 1):
        expose_acquaintance_gates(strategy)
        replace_acquaintance_with_swap_network(strategy, qubit_order, size_to_acquaint, swap_gate)
    return strategy