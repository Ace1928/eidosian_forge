import logging
from copy import deepcopy
import time
import rustworkx
from qiskit.circuit import SwitchCaseOp, ControlFlowOp, Clbit, ClassicalRegister
from qiskit.circuit.library.standard_gates import SwapGate
from qiskit.circuit.controlflow import condition_resources, node_resources
from qiskit.converters import dag_to_circuit
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.coupling import CouplingMap
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.layout import Layout
from qiskit.transpiler.target import Target
from qiskit.transpiler.passes.layout import disjoint_utils
from qiskit.dagcircuit import DAGCircuit
from qiskit.utils.parallel import CPU_COUNT
from qiskit._accelerate.sabre_swap import (
from qiskit._accelerate.nlayout import NLayout
def process_dag(block_dag, wire_map):
    dag_list = []
    node_blocks = {}
    for node in block_dag.topological_op_nodes():
        cargs_bits = set(node.cargs)
        if node.op.condition is not None:
            cargs_bits.update(condition_resources(node.op.condition).clbits)
        if isinstance(node.op, SwitchCaseOp):
            target = node.op.target
            if isinstance(target, Clbit):
                cargs_bits.add(target)
            elif isinstance(target, ClassicalRegister):
                cargs_bits.update(target)
            else:
                cargs_bits.update(node_resources(target).clbits)
        cargs = {block_dag.find_bit(x).index for x in cargs_bits}
        if isinstance(node.op, ControlFlowOp):
            node_blocks[node._node_id] = [recurse(block, {inner: wire_map[outer] for inner, outer in zip(block.qubits, node.qargs)}) for block in node.op.blocks]
        dag_list.append((node._node_id, [wire_map[x] for x in node.qargs], cargs, getattr(node.op, '_directive', False)))
    return SabreDAG(num_physical_qubits, block_dag.num_clbits(), dag_list, node_blocks)