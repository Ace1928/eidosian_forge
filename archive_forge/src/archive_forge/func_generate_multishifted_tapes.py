import functools
import itertools
import numbers
import warnings
import numpy as np
from scipy.linalg import solve as linalg_solve
import pennylane as qml
from pennylane.measurements import MeasurementProcess
from pennylane.ops.functions import bind_new_parameters
from pennylane.tape import QuantumScript
def generate_multishifted_tapes(tape, indices, shifts, multipliers=None):
    """Generate a list of tapes where multiple marked trainable
    parameters have been shifted by the provided shift values.

    Args:
        tape (.QuantumTape): input quantum tape
        indices (Sequence[int]): indices of the trainable parameters to shift
        shifts (Sequence[Sequence[float or int]]): Nested sequence of shift values.
            The length of the outer Sequence determines how many parameter-shifted
            tapes are created. The lengths of the inner sequences should match and
            have the same length as ``indices``.
        multipliers (Sequence[Sequence[float or int]]): Nested sequence
            of multiplier values of the same format as `shifts``. Each multiplier
            scales the corresponding gate parameter before the shift is applied.
            If not provided, the parameters will not be scaled.

    Returns:
        list[QuantumTape]: List of quantum tapes. Each tape has the marked parameters
            indicated by ``indices`` shifted by the values of ``shifts``. The number
            of tapes will match the summed lengths of all inner sequences in ``shifts``
            and ``multipliers`` (if provided).
    """
    if multipliers is None:
        multipliers = np.ones_like(shifts)
    tapes = [_copy_and_shift_params(tape, indices, _shifts, _multipliers, cast=True) for _shifts, _multipliers in zip(shifts, multipliers)]
    return tapes