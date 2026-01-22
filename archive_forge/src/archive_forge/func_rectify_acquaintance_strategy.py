import collections
from typing import cast, Dict, List, Optional, Sequence, Union, TYPE_CHECKING
from cirq import circuits, ops, transformers
from cirq.contrib.acquaintance.gates import SwapNetworkGate, AcquaintanceOpportunityGate
from cirq.contrib.acquaintance.devices import get_acquaintance_size
from cirq.contrib.acquaintance.permutation import PermutationGate
def rectify_acquaintance_strategy(circuit: 'cirq.Circuit', acquaint_first: bool=True) -> None:
    """Splits moments so that they contain either only acquaintance or permutation gates.

    Orders resulting moments so that the first one is of the same type as the previous one.

    Args:
        circuit: The acquaintance strategy to rectify.
        acquaint_first: Whether to make acquaintance moment first in when
        splitting the first mixed moment.

    Raises:
        TypeError: If the circuit is not an acquaintance strategy.
    """
    rectified_moments = []
    for moment in circuit:
        gate_type_to_ops: Dict[bool, List[ops.GateOperation]] = collections.defaultdict(list)
        for op in moment.operations:
            gate_op = cast(ops.GateOperation, op)
            is_acquaintance = isinstance(gate_op.gate, AcquaintanceOpportunityGate)
            gate_type_to_ops[is_acquaintance].append(gate_op)
        if len(gate_type_to_ops) == 1:
            rectified_moments.append(moment)
            continue
        for acquaint_first in sorted(gate_type_to_ops.keys(), reverse=acquaint_first):
            rectified_moments.append(circuits.Moment(gate_type_to_ops[acquaint_first]))
    circuit._moments = rectified_moments