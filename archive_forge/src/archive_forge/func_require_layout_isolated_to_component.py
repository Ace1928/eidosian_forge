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
def require_layout_isolated_to_component(dag: DAGCircuit, components_source: Union[Target, CouplingMap]):
    """
    Check that the layout of the dag does not require connectivity across connected components
    in the CouplingMap

    Args:
        dag: DAGCircuit to check.
        components_source: Target to check against.

    Raises:
        TranspilerError: Chosen layout is not valid for the target disjoint connectivity.
    """
    if isinstance(components_source, Target):
        coupling_map = components_source.build_coupling_map(filter_idle_qubits=True)
    else:
        coupling_map = components_source
    component_sets = [set(x.graph.nodes()) for x in coupling_map.connected_components()]
    for inst in dag.two_qubit_ops():
        component_index = None
        for i, component_set in enumerate(component_sets):
            if dag.find_bit(inst.qargs[0]).index in component_set:
                component_index = i
                break
        if dag.find_bit(inst.qargs[1]).index not in component_sets[component_index]:
            raise TranspilerError(f'The circuit has an invalid layout as two qubits need to interact in disconnected components of the coupling map. The physical qubit {dag.find_bit(inst.qargs[1]).index} needs to interact with the qubit {dag.find_bit(inst.qargs[0]).index} and they belong to different components')