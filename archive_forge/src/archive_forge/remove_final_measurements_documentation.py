from qiskit.transpiler.basepasses import TransformationPass
from qiskit.dagcircuit import DAGOpNode
Run the RemoveFinalMeasurements pass on `dag`.

        Args:
            dag (DAGCircuit): the DAG to be optimized.

        Returns:
            DAGCircuit: the optimized DAG.
        