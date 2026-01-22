from qiskit.exceptions import QiskitError
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passes.utils import control_flow
from qiskit.circuit import ControlledGate, ControlFlowOp
from qiskit.converters.circuit_to_dag import circuit_to_dag
Run the UnrollCustomDefinitions pass on `dag`.

        Args:
            dag (DAGCircuit): input dag

        Raises:
            QiskitError: if unable to unroll given the basis due to undefined
            decomposition rules (such as a bad basis) or excessive recursion.

        Returns:
            DAGCircuit: output unrolled dag
        