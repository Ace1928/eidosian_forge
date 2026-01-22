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
def separate_dag(dag: DAGCircuit) -> List[DAGCircuit]:
    """Separate a dag circuit into it's connected components."""
    split_barriers(dag)
    im_graph, _, qubit_map, __ = vf2_utils.build_interaction_graph(dag)
    connected_components = rx.weakly_connected_components(im_graph)
    component_qubits = []
    for component in connected_components:
        component_qubits.append({qubit_map[x] for x in component})
    qubits = set(dag.qubits)
    decomposed_dags = []
    for dag_qubits in component_qubits:
        new_dag = dag.copy_empty_like()
        new_dag.remove_qubits(*qubits - dag_qubits)
        new_dag.global_phase = 0
        for node in dag.topological_op_nodes():
            if dag_qubits.issuperset(node.qargs):
                new_dag.apply_operation_back(node.op, node.qargs, node.cargs, check=False)
        idle_clbits = []
        for bit, node in new_dag.input_map.items():
            succ_node = next(new_dag.successors(node))
            if isinstance(succ_node, DAGOutNode) and isinstance(succ_node.wire, Clbit):
                idle_clbits.append(bit)
        new_dag.remove_clbits(*idle_clbits)
        combine_barriers(new_dag)
        decomposed_dags.append(new_dag)
    combine_barriers(dag, retain_uuid=False)
    return decomposed_dags