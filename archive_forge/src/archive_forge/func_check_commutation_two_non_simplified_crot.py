import numpy as np
import pennylane as qml
from pennylane.pauli.utils import is_pauli_word, pauli_to_binary, _wire_map_from_pauli_pair
def check_commutation_two_non_simplified_crot(operation1, operation2):
    """Check commutation for two CRot that were not simplified.

    Args:
        operation1 (pennylane.Operation): First operation.
        operation2 (pennylane.Operation): Second operation.

    Returns:
         Bool: True if commutation, False otherwise.
    """
    target_wires_1 = qml.wires.Wires([w for w in operation1.wires if w not in operation1.control_wires])
    target_wires_2 = qml.wires.Wires([w for w in operation2.wires if w not in operation2.control_wires])
    control_control = intersection(operation1.control_wires, operation2.control_wires)
    target_target = intersection(target_wires_1, target_wires_2)
    if control_control:
        if target_target:
            return _check_mat_commutation(operation1, operation2)
        return True
    if target_target:
        return _check_mat_commutation(qml.Rot(*operation1.data, wires=operation1.wires[1]), qml.Rot(*operation2.data, wires=operation2.wires[1]))
    return False