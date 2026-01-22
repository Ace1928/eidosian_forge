import numpy as np
import pennylane as qml
from pennylane.pauli.utils import is_pauli_word, pauli_to_binary, _wire_map_from_pauli_pair
def check_commutation_two_non_simplified_rotations(operation1, operation2):
    """Check that the operations are two non simplified operations. If it is the case, then it checks commutation
    for two rotations that were not simplified.

    Only allowed ops are `U2`, `U3`, `Rot`, `CRot`.

    Args:
        operation1 (pennylane.Operation): First operation.
        operation2 (pennylane.Operation): Second operation.

    Returns:
         Bool: True if commutation, False otherwise, None if not two rotations.
    """
    target_wires_1 = qml.wires.Wires([w for w in operation1.wires if w not in operation1.control_wires])
    target_wires_2 = qml.wires.Wires([w for w in operation2.wires if w not in operation2.control_wires])
    if operation1.name == 'CRot':
        if intersection(target_wires_1, operation2.wires):
            op1_rot = qml.Rot(*operation1.data, wires=target_wires_1)
            return _check_mat_commutation(op1_rot, operation2)
        return _commutes(operation2.name, 'ctrl')
    if operation2.name == 'CRot':
        if intersection(target_wires_2, operation1.wires):
            op2_rot = qml.Rot(*operation2.data, wires=target_wires_2)
            return _check_mat_commutation(op2_rot, operation1)
        return _commutes(operation1.name, 'ctrl')
    return _check_mat_commutation(operation1, operation2)