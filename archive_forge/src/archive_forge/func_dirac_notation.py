from typing import Any, cast, Iterable, List, Optional, Sequence, Set, TYPE_CHECKING, Tuple, Union
import itertools
import numpy as np
from cirq import value
from cirq._doc import document
def dirac_notation(state_vector: np.ndarray, decimals: int=2, qid_shape: Optional[Tuple[int, ...]]=None) -> str:
    """Returns the state vector as a string in Dirac notation.

    For example:

    >>> state_vector = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=np.complex64)
    >>> print(cirq.dirac_notation(state_vector))
    0.71|0⟩ + 0.71|1⟩


    Args:
        state_vector: A sequence representing a state vector in which
            the ordering mapping to qubits follows the standard Kronecker
            convention of numpy.kron (big-endian).
        decimals: How many decimals to include in the pretty print.
        qid_shape: specifies the dimensions of the qudits for the input
            `state_vector`.  If not specified, qubits are assumed and the
            `state_vector` must have a dimension a power of two.

    Returns:
        A pretty string consisting of a sum of computational basis kets
        and non-zero floats of the specified accuracy.

    Raises:
        ValueError: If there is a shape mismatch between state_vector and qid_shape.
            Otherwise, when qid_shape is not mentioned and length of state_vector
            is not a power of 2.
    """
    if qid_shape is None:
        qid_shape = (2,) * (len(state_vector).bit_length() - 1)
    if len(state_vector) != np.prod(qid_shape, dtype=np.int64):
        raise ValueError(f'state_vector has incorrect size. Expected {np.prod(qid_shape, dtype=np.int64)} but was {len(state_vector)}.')
    digit_separator = '' if max(qid_shape, default=0) < 10 else ','
    perm_list = [digit_separator.join(seq) for seq in itertools.product(*((str(i) for i in range(d)) for d in qid_shape))]
    components = []
    ket = '|{}⟩'
    for x in range(len(perm_list)):
        format_str = '({:.' + str(decimals) + 'g})'
        val = round(state_vector[x].real, decimals) + 1j * round(state_vector[x].imag, decimals)
        if round(val.real, decimals) == 0 and round(val.imag, decimals) != 0:
            val = val.imag
            format_str = '{:.' + str(decimals) + 'g}j'
        elif round(val.imag, decimals) == 0 and round(val.real, decimals) != 0:
            val = val.real
            format_str = '{:.' + str(decimals) + 'g}'
        if val != 0:
            if round(state_vector[x].real, decimals) == 1 and round(state_vector[x].imag, decimals) == 0:
                components.append(ket.format(perm_list[x]))
            else:
                components.append((format_str + ket).format(val, perm_list[x]))
    if not components:
        return '0'
    return ' + '.join(components).replace(' + -', ' - ')