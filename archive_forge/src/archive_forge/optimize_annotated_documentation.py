from typing import Optional, List, Tuple
from qiskit.circuit.controlflow import CONTROL_FLOW_OP_NAMES
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.circuit.annotated_operation import AnnotatedOperation, _canonicalize_modifiers
from qiskit.circuit import EquivalenceLibrary, ControlledGate, Operation, ControlFlowOp
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passes.utils import control_flow
from qiskit.transpiler.target import Target
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.exceptions import TranspilerError

        Recursively handles gate definitions.
        Returns True if did something.
        