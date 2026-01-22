import numpy as np
import pennylane as qml
from pennylane.pauli.utils import is_pauli_word, pauli_to_binary, _wire_map_from_pauli_pair
def commutes_inner(op_name1, op_name2):
    """Determine whether or not two operations commute.

        Relies on ``commutation_map`` from the enclosing namespace of ``_create_commute_function``.

        Args:
            op_name1 (str): name of one operation
            op_name2 (str): name of the second operation

        Returns:
            Bool

        """
    return op_name1 in commutation_map[op_name2]