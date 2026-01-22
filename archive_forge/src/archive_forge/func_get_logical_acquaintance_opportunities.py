from typing import FrozenSet, Sequence, Set, TYPE_CHECKING
from cirq import devices
from cirq.contrib.acquaintance.executor import AcquaintanceOperation, ExecutionStrategy
from cirq.contrib.acquaintance.mutation_utils import expose_acquaintance_gates
from cirq.contrib.acquaintance.permutation import LogicalIndex, LogicalMapping
from cirq.contrib import circuitdag
def get_logical_acquaintance_opportunities(strategy: 'cirq.Circuit', initial_mapping: LogicalMapping) -> Set[FrozenSet[LogicalIndex]]:
    acquaintance_dag = get_acquaintance_dag(strategy, initial_mapping)
    logical_acquaintance_opportunities = set()
    for op in acquaintance_dag.all_operations():
        logical_acquaintance_opportunities.add(frozenset(op.logical_indices))
    return logical_acquaintance_opportunities