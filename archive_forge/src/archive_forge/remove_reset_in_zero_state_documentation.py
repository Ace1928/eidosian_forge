from qiskit.circuit import Reset
from qiskit.dagcircuit import DAGInNode
from qiskit.transpiler.basepasses import TransformationPass
Run the RemoveResetInZeroState pass on `dag`.

        Args:
            dag (DAGCircuit): the DAG to be optimized.

        Returns:
            DAGCircuit: the optimized DAG.
        