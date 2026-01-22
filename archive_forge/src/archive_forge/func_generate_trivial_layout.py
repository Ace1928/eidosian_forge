from __future__ import annotations
from typing import List
from dataclasses import dataclass
from qiskit import circuit
from qiskit.circuit.quantumregister import Qubit, QuantumRegister
from qiskit.transpiler.exceptions import LayoutError
from qiskit.converters import isinstanceint
@staticmethod
def generate_trivial_layout(*regs):
    """Creates a trivial ("one-to-one") Layout with the registers and qubits in `regs`.

        Args:
            *regs (Registers, Qubits): registers and qubits to include in the layout.
        Returns:
            Layout: A layout with all the `regs` in the given order.
        """
    layout = Layout()
    for reg in regs:
        if isinstance(reg, QuantumRegister):
            layout.add_register(reg)
        else:
            layout.add(reg)
    return layout