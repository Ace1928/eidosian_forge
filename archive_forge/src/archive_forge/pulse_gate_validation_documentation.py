from qiskit.dagcircuit import DAGCircuit
from qiskit.pulse import Play
from qiskit.transpiler.basepasses import AnalysisPass
from qiskit.transpiler.exceptions import TranspilerError
Run the pulse gate validation attached to ``dag``.

        Args:
            dag: DAG to be validated.

        Returns:
            DAGCircuit: DAG with consistent timing and op nodes annotated with duration.

        Raises:
            TranspilerError: When pulse gate violate pulse controller constraints.
        