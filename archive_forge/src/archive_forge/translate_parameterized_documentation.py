from __future__ import annotations
from qiskit.circuit import Instruction, ParameterExpression, Qubit, Clbit
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.circuit.equivalence_library import EquivalenceLibrary
from qiskit.exceptions import QiskitError
from qiskit.transpiler import Target
from qiskit.transpiler.basepasses import TransformationPass
from .basis_translator import BasisTranslator
Run the transpiler pass.

        Args:
            dag: The DAG circuit in which the parameterized gates should be unrolled.

        Returns:
            A DAG where the parameterized gates have been unrolled.

        Raises:
            QiskitError: If the circuit cannot be unrolled.
        