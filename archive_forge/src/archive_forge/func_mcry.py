from math import pi
from typing import Optional, Union, Tuple, List
import numpy as np
from qiskit.circuit import QuantumCircuit, QuantumRegister, Qubit
from qiskit.circuit.library.standard_gates.x import MCXGate
from qiskit.circuit.library.standard_gates.u3 import _generate_gray_code
from qiskit.circuit.parameterexpression import ParameterValueType
from qiskit.exceptions import QiskitError
def mcry(self, theta: ParameterValueType, q_controls: Union[QuantumRegister, List[Qubit]], q_target: Qubit, q_ancillae: Optional[Union[QuantumRegister, Tuple[QuantumRegister, int]]]=None, mode: str=None, use_basis_gates=False):
    """
    Apply Multiple-Controlled Y rotation gate

    Args:
        self (QuantumCircuit): The QuantumCircuit object to apply the mcry gate on.
        theta (float): angle theta
        q_controls (list(Qubit)): The list of control qubits
        q_target (Qubit): The target qubit
        q_ancillae (QuantumRegister or tuple(QuantumRegister, int)): The list of ancillary qubits.
        mode (string): The implementation mode to use
        use_basis_gates (bool): use p, u, cx

    Raises:
        QiskitError: parameter errors
    """
    from .ry import RYGate
    control_qubits = self.qbit_argument_conversion(q_controls)
    target_qubit = self.qbit_argument_conversion(q_target)
    if len(target_qubit) != 1:
        raise QiskitError('The mcrz gate needs a single qubit as target.')
    ancillary_qubits = [] if q_ancillae is None else self.qbit_argument_conversion(q_ancillae)
    all_qubits = control_qubits + target_qubit + ancillary_qubits
    target_qubit = target_qubit[0]
    self._check_dups(all_qubits)
    if mode is None:
        additional_vchain = MCXGate.get_num_ancilla_qubits(len(control_qubits), 'v-chain')
        if len(ancillary_qubits) >= additional_vchain:
            mode = 'basic'
        else:
            mode = 'noancilla'
    if mode == 'basic':
        self.ry(theta / 2, q_target)
        self.mcx(q_controls, q_target, q_ancillae, mode='v-chain')
        self.ry(-theta / 2, q_target)
        self.mcx(q_controls, q_target, q_ancillae, mode='v-chain')
    elif mode == 'noancilla':
        n_c = len(control_qubits)
        if n_c == 1:
            _apply_cu(self, theta, 0, 0, control_qubits[0], target_qubit, use_basis_gates=use_basis_gates)
        elif n_c < 4:
            theta_step = theta * (1 / 2 ** (n_c - 1))
            _apply_mcu_graycode(self, theta_step, 0, 0, control_qubits, target_qubit, use_basis_gates=use_basis_gates)
        else:
            cgate = _mcsu2_real_diagonal(RYGate(theta).to_matrix(), num_controls=len(control_qubits), use_basis_gates=use_basis_gates)
            self.compose(cgate, control_qubits + [target_qubit], inplace=True)
    else:
        raise QiskitError(f'Unrecognized mode for building MCRY circuit: {mode}.')