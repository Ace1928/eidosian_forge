from __future__ import annotations
import numpy as np
from qiskit.transpiler import Layout, CouplingMap
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.passes.routing.algorithms import ApproximateTokenSwapper
from qiskit.transpiler.target import Target
Apply the specified partial permutation to the circuit.

        Args:
            dag (DAGCircuit): DAG to transform the layout of.

        Returns:
            DAGCircuit: The DAG with transformed layout.

        Raises:
            TranspilerError: if the coupling map or the layout are not compatible with the DAG.
                Or if either of string from/to_layout is not found in `property_set`.
        