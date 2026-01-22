from typing import Optional, Union, List, Tuple
import rustworkx as rx
from qiskit.circuit.operation import Operation
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit import ControlFlowOp, ControlledGate, EquivalenceLibrary
from qiskit.transpiler.passes.utils import control_flow
from qiskit.transpiler.target import Target
from qiskit.transpiler.coupling import CouplingMap
from qiskit.dagcircuit.dagcircuit import DAGCircuit
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.passes.routing.algorithms import ApproximateTokenSwapper
from qiskit.circuit.annotated_operation import (
from qiskit.synthesis.clifford import (
from qiskit.synthesis.linear import synth_cnot_count_full_pmh, synth_cnot_depth_line_kms
from qiskit.synthesis.permutation import (
from .plugin import HighLevelSynthesisPluginManager, HighLevelSynthesisPlugin
def set_methods(self, hls_name, hls_methods):
    """Sets the list of synthesis methods for a given higher-level-object. This overwrites
        the lists of methods if also set previously."""
    self.methods[hls_name] = hls_methods