from qiskit.circuit import Qubit
from qiskit.circuit.delay import Delay
from qiskit.dagcircuit import DAGCircuit, DAGNode, DAGOutNode
from qiskit.transpiler.target import Target
from .base_padding import BasePadding
Create new padding delay pass.

        Args:
            fill_very_end: Set ``True`` to fill the end of circuit with delay.
            target: The :class:`~.Target` representing the target backend.
                If it is supplied and does not support delay instruction on a qubit,
                padding passes do not pad any idle time of the qubit.
        