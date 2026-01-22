from __future__ import annotations
from collections import defaultdict
from typing import List, Callable, TypeVar, Dict, Union
import uuid
import rustworkx as rx
from qiskit.dagcircuit import DAGOpNode
from qiskit.circuit import Qubit, Barrier, Clbit
from qiskit.dagcircuit.dagcircuit import DAGCircuit
from qiskit.dagcircuit.dagnode import DAGOutNode
from qiskit.transpiler.coupling import CouplingMap
from qiskit.transpiler.target import Target
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.passes.layout import vf2_utils
def combine_barriers(dag: DAGCircuit, retain_uuid: bool=True):
    """Mutate input dag to combine barriers with UUID labels into a single barrier."""
    qubit_indices = {bit: index for index, bit in enumerate(dag.qubits)}
    uuid_map: dict[uuid.UUID, DAGOpNode] = {}
    for node in dag.op_nodes(Barrier):
        if node.op.label:
            if '_uuid=' in node.op.label:
                barrier_uuid = node.op.label
            else:
                continue
            if barrier_uuid in uuid_map:
                other_node = uuid_map[barrier_uuid]
                num_qubits = len(other_node.qargs) + len(node.qargs)
                new_op = Barrier(num_qubits, label=barrier_uuid)
                new_node = dag.replace_block_with_op([node, other_node], new_op, qubit_indices)
                uuid_map[barrier_uuid] = new_node
            else:
                uuid_map[barrier_uuid] = node
    if not retain_uuid:
        for node in dag.op_nodes(Barrier):
            if isinstance(node.op.label, str) and node.op.label.startswith('_none_uuid='):
                node.op.label = None
            elif isinstance(node.op.label, str) and '_uuid=' in node.op.label:
                original_label = '_uuid='.join(node.op.label.split('_uuid=')[:-1])
                node.op.label = original_label