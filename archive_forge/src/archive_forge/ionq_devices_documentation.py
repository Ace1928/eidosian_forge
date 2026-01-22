from typing import Sequence
from typing import Union
import cirq
from cirq_ionq import ionq_gateset
Construct the device.

        Args:
            qubits: The qubits upon which this device acts or the number of qubits. If the number
                of qubits, then the qubits will be `cirq.LineQubit`s from 0 to this number minus
                one.
            atol: The absolute tolerance used for gate calculations and decompositions.
        