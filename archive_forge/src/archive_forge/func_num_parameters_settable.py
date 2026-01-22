from __future__ import annotations
import typing
from collections.abc import Callable, Mapping, Sequence
from itertools import combinations
import numpy
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit import Instruction, Parameter, ParameterVector, ParameterExpression
from qiskit.exceptions import QiskitError
from ..blueprintcircuit import BlueprintCircuit
@property
def num_parameters_settable(self) -> int:
    """The number of total parameters that can be set to distinct values.

        This does not change when the parameters are bound or exchanged for same parameters,
        and therefore is different from ``num_parameters`` which counts the number of unique
        :class:`~qiskit.circuit.Parameter` objects currently in the circuit.

        Returns:
            The number of parameters originally available in the circuit.

        Note:
            This quantity does not require the circuit to be built yet.
        """
    num = 0
    for i in range(self._reps):
        for j, block in enumerate(self.entanglement_blocks):
            entangler_map = self.get_entangler_map(i, j, block.num_qubits)
            num += len(entangler_map) * len(get_parameters(block))
    if self._skip_unentangled_qubits:
        unentangled_qubits = self.get_unentangled_qubits()
    num_rot = 0
    for block in self.rotation_blocks:
        block_indices = [list(range(j * block.num_qubits, (j + 1) * block.num_qubits)) for j in range(self.num_qubits // block.num_qubits)]
        if self._skip_unentangled_qubits:
            block_indices = [indices for indices in block_indices if set(indices).isdisjoint(unentangled_qubits)]
        num_rot += len(block_indices) * len(get_parameters(block))
    num += num_rot * (self._reps + int(not self._skip_final_rotation_layer))
    return num