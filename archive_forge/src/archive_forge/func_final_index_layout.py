from __future__ import annotations
from typing import List
from dataclasses import dataclass
from qiskit import circuit
from qiskit.circuit.quantumregister import Qubit, QuantumRegister
from qiskit.transpiler.exceptions import LayoutError
from qiskit.converters import isinstanceint
def final_index_layout(self, filter_ancillas: bool=True) -> List[int]:
    """Generate the final layout as an array of integers

        This method will generate an array of final positions for each qubit in the output circuit.
        For example, if you had an input circuit like::

            qc = QuantumCircuit(3)
            qc.h(0)
            qc.cx(0, 1)
            qc.cx(0, 2)

        and the output from the transpiler was::

            tqc = QuantumCircuit(3)
            qc.h(2)
            qc.cx(2, 1)
            qc.swap(0, 1)
            qc.cx(2, 1)

        then the return from this function would be a list of::

            [2, 0, 1]

        because qubit 0 in the original circuit's final state is on qubit 3 in the output circuit,
        qubit 1 in the original circuit's final state is on qubit 0, and qubit 2's final state is
        on qubit. The output list length will be as wide as the input circuit's number of qubits,
        as the output list from this method is for tracking the permutation of qubits in the
        original circuit caused by the transpiler.

        Args:
            filter_ancillas: If set to ``False`` any ancillas allocated in the output circuit will be
                included in the layout.

        Returns:
            A list of final positions for each input circuit qubit
        """
    if self._input_qubit_count is None:
        num_source_qubits = len([x for x in self.input_qubit_mapping if getattr(x, '_register', '').startswith('ancilla')])
    else:
        num_source_qubits = self._input_qubit_count
    if self._output_qubit_list is None:
        circuit_qubits = list(self.final_layout.get_virtual_bits())
    else:
        circuit_qubits = self._output_qubit_list
    pos_to_virt = {v: k for k, v in self.input_qubit_mapping.items()}
    qubit_indices = []
    if filter_ancillas:
        num_qubits = num_source_qubits
    else:
        num_qubits = len(self._output_qubit_list)
    for index in range(num_qubits):
        qubit_idx = self.initial_layout[pos_to_virt[index]]
        if self.final_layout is not None:
            qubit_idx = self.final_layout[circuit_qubits[qubit_idx]]
        qubit_indices.append(qubit_idx)
    return qubit_indices