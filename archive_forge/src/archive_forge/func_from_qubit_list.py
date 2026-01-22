from __future__ import annotations
from typing import List
from dataclasses import dataclass
from qiskit import circuit
from qiskit.circuit.quantumregister import Qubit, QuantumRegister
from qiskit.transpiler.exceptions import LayoutError
from qiskit.converters import isinstanceint
@staticmethod
def from_qubit_list(qubit_list, *qregs):
    """
        Populates a Layout from a list containing virtual
        qubits, Qubit or None.

        Args:
            qubit_list (list):
                e.g.: [qr[0], None, qr[2], qr[3]]
            *qregs (QuantumRegisters): The quantum registers to apply
                the layout to.
        Returns:
            Layout: the corresponding Layout object
        Raises:
            LayoutError: If the elements are not Qubit or None
        """
    out = Layout()
    for physical, virtual in enumerate(qubit_list):
        if virtual is None:
            continue
        if isinstance(virtual, Qubit):
            if virtual in out._v2p:
                raise LayoutError('Duplicate values not permitted; Layout is bijective.')
            out[virtual] = physical
        else:
            raise LayoutError('The list should contain elements of the Bits or NoneTypes')
    for qreg in qregs:
        out.add_register(qreg)
    return out