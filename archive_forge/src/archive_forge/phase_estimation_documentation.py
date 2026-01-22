from typing import Optional
from qiskit.circuit import QuantumCircuit, QuantumRegister
from .basis_change import QFT

        Args:
            num_evaluation_qubits: The number of evaluation qubits.
            unitary: The unitary operation :math:`U` which will be repeated and controlled.
            iqft: A inverse Quantum Fourier Transform, per default the inverse of
                :class:`~qiskit.circuit.library.QFT` is used. Note that the QFT should not include
                the usual swaps!
            name: The name of the circuit.

        .. note::

            The inverse QFT should not include a swap of the qubit order.

        Reference Circuit:
            .. plot::

               from qiskit.circuit import QuantumCircuit
               from qiskit.circuit.library import PhaseEstimation
               from qiskit.visualization.library import _generate_circuit_library_visualization
               unitary = QuantumCircuit(2)
               unitary.x(0)
               unitary.y(1)
               circuit = PhaseEstimation(3, unitary)
               _generate_circuit_library_visualization(circuit)
        