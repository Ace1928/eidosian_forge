import numpy as np
from qiskit.exceptions import QiskitError
def format_statevector(vec, decimals=None):
    """Format statevector coming from the backend to present to the Qiskit user.

    Args:
        vec (list): a list of [re, im] complex numbers.
        decimals (int): the number of decimals in the statevector.
            If None, no rounding is done.

    Returns:
        list[complex]: a list of python complex numbers.
    """
    from qiskit.quantum_info.states.statevector import Statevector
    if isinstance(vec, Statevector):
        if decimals:
            return Statevector(np.around(vec.data, decimals=decimals), dims=vec.dims())
        return vec
    if isinstance(vec, np.ndarray):
        if decimals:
            return np.around(vec, decimals=decimals)
        return vec
    num_basis = len(vec)
    if vec and isinstance(vec[0], complex):
        vec_complex = np.array(vec, dtype=complex)
    else:
        vec_complex = np.zeros(num_basis, dtype=complex)
        for i in range(num_basis):
            vec_complex[i] = vec[i][0] + 1j * vec[i][1]
    if decimals:
        vec_complex = np.around(vec_complex, decimals=decimals)
    return vec_complex