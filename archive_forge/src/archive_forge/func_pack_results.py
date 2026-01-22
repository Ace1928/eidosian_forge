import json
from typing import Any, cast, Dict, Optional, Sequence, Tuple, TYPE_CHECKING, Iterator
import numpy as np
import sympy
import cirq
from cirq_google.api.v1 import operations_pb2
def pack_results(measurements: Sequence[Tuple[str, np.ndarray]]) -> bytes:
    """Pack measurement results into a byte string.

    Args:
        measurements: A sequence of tuples, one for each measurement, consisting
            of a string key and an array of boolean data. The data should be
            a 2-D array indexed by (repetition, qubit_index). All data for all
            measurements must have the same number of repetitions.

    Returns:
        Packed bytes, as described in the unpack_results docstring below.

    Raises:
        ValueError: If the measurement data do not have the compatible shapes.
    """
    if not measurements:
        return b''
    shapes = [(key, np.shape(data)) for key, data in measurements]
    if not all((len(shape) == 2 for _, shape in shapes)):
        raise ValueError(f'Expected 2-D data: shapes={shapes}')
    reps = shapes[0][1][0]
    if not all((shape[0] == reps for _, shape in shapes)):
        raise ValueError(f'Expected same reps for all keys: shapes={shapes}')
    bits = np.hstack([np.asarray(data, dtype=bool) for _, data in measurements])
    bits = bits.reshape(-1)
    remainder = len(bits) % 8
    if remainder:
        bits = np.pad(bits, (0, 8 - remainder), 'constant')
    bits = bits.reshape((-1, 8))[:, ::-1]
    byte_arr = np.packbits(bits, axis=1).reshape(-1)
    return byte_arr.tobytes()