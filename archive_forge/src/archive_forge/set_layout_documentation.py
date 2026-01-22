from qiskit.transpiler import Layout
from qiskit.transpiler.exceptions import InvalidLayoutError
from qiskit.transpiler.basepasses import AnalysisPass
Run the SetLayout pass on ``dag``.

        Args:
            dag (DAGCircuit): DAG to map.

        Returns:
            DAGCircuit: the original DAG.
        