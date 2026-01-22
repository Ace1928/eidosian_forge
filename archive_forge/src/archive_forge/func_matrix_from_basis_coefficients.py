from typing import Dict, Tuple
import numpy as np
from cirq import value
from cirq._doc import document
def matrix_from_basis_coefficients(expansion: value.LinearDict[str], basis: Dict[str, np.ndarray]) -> np.ndarray:
    """Computes linear combination of basis vectors with given coefficients."""
    some_element = next(iter(basis.values()))
    result = np.zeros_like(some_element, dtype=np.complex128)
    for name, coefficient in expansion.items():
        result += coefficient * basis[name]
    return result