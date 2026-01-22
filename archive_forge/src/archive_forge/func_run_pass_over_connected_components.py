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
def run_pass_over_connected_components(dag: DAGCircuit, components_source: Union[Target, CouplingMap], run_func: Callable[[DAGCircuit, CouplingMap], T]) -> List[T]:
    """Run a transpiler pass inner function over mapped components."""
    if isinstance(components_source, Target):
        coupling_map = components_source.build_coupling_map(filter_idle_qubits=True)
    else:
        coupling_map = components_source
    cmap_components = coupling_map.connected_components()
    if len(cmap_components) == 1:
        if dag.num_qubits() > cmap_components[0].size():
            raise TranspilerError('A connected component of the DAGCircuit is too large for any of the connected components in the coupling map.')
        return [run_func(dag, cmap_components[0])]
    dag_components = separate_dag(dag)
    mapped_components = map_components(dag_components, cmap_components)
    out_component_pairs = []
    for cmap_index, dags in mapped_components.items():
        out_dag = dag_components[dags.pop()]
        for dag_index in dags:
            dag = dag_components[dag_index]
            out_dag.add_qubits(dag.qubits)
            out_dag.add_clbits(dag.clbits)
            for qreg in dag.qregs:
                out_dag.add_qreg(qreg)
            for creg in dag.cregs:
                out_dag.add_creg(creg)
            out_dag.compose(dag, qubits=dag.qubits, clbits=dag.clbits)
        out_component_pairs.append((out_dag, cmap_components[cmap_index]))
    res = [run_func(out_dag, cmap) for out_dag, cmap in out_component_pairs]
    return res