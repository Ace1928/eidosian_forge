from typing import Any, Dict, List, Optional, Set, Sequence, Tuple, TYPE_CHECKING
import itertools
import networkx as nx
from cirq import circuits, ops, protocols
from cirq.transformers import transformer_api, transformer_primitives
from cirq.transformers.routing import mapping_manager, line_initial_mapper
def process_executable_two_qubit_ops(timestep: int) -> int:
    unexecutable_ops: List['cirq.Operation'] = []
    unexecutable_ops_ints: List[QidIntPair] = []
    for op, op_ints in zip(two_qubit_ops[timestep], two_qubit_ops_ints[timestep]):
        if mm.is_adjacent(*op_ints):
            routed_ops[timestep].append(mm.mapped_op(op))
        else:
            unexecutable_ops.append(op)
            unexecutable_ops_ints.append(op_ints)
    two_qubit_ops[timestep] = unexecutable_ops
    two_qubit_ops_ints[timestep] = unexecutable_ops_ints
    return len(unexecutable_ops)