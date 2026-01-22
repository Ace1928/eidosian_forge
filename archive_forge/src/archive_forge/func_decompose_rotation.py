from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING, Union
import numpy as np
from cirq import protocols, value, linalg, qis
from cirq._import import LazyLoader
from cirq.ops import common_gates, named_qubit, raw_types, pauli_gates, phased_x_z_gate
from cirq.ops.pauli_gates import Pauli
from cirq.type_workarounds import NotImplementedType
def decompose_rotation(self) -> Sequence[Tuple[Pauli, int]]:
    """Decomposes this clifford into a series of pauli rotations.

        Each rotation is given as a tuple of (axis, quarter_turns),
        where axis is a Pauli giving the axis to rotate about. The
        result will be a sequence of zero, one, or two rotations.

        Note that the combined unitary effect of these rotations may
        differ from cirq.unitary(self) by a global phase.
        """
    x_rot = self.pauli_tuple(pauli_gates.X)
    y_rot = self.pauli_tuple(pauli_gates.Y)
    z_rot = self.pauli_tuple(pauli_gates.Z)
    whole_arr = (x_rot[0] == pauli_gates.X, y_rot[0] == pauli_gates.Y, z_rot[0] == pauli_gates.Z)
    num_whole = sum(whole_arr)
    flip_arr = (x_rot[1], y_rot[1], z_rot[1])
    num_flip = sum(flip_arr)
    if num_whole == 3:
        if num_flip == 0:
            return []
        pauli = Pauli.by_index(flip_arr.index(False))
        return [(pauli, 2)]
    if num_whole == 1:
        index = whole_arr.index(True)
        pauli = Pauli.by_index(index)
        next_pauli = Pauli.by_index(index + 1)
        flip = flip_arr[index]
        output = []
        if flip:
            output.append((next_pauli, 2))
        if self.pauli_tuple(next_pauli)[1]:
            output.append((pauli, -1))
        else:
            output.append((pauli, 1))
        return output
    elif num_whole == 0:
        if x_rot[0] == pauli_gates.Y:
            return [(pauli_gates.X, -1 if y_rot[1] else 1), (pauli_gates.Z, -1 if x_rot[1] else 1)]
        return [(pauli_gates.Z, 1 if y_rot[1] else -1), (pauli_gates.X, 1 if z_rot[1] else -1)]
    assert False, 'Impossible condition where this gate only rotates one Pauli to a different Pauli.'