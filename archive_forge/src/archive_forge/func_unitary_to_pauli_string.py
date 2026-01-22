from typing import Optional, Tuple, cast
import numpy as np
import numpy.typing as npt
from cirq.ops import DensePauliString
from cirq import protocols
def unitary_to_pauli_string(U: npt.NDArray, eps: float=1e-15) -> Optional[DensePauliString]:
    """Attempts to find a pauli string (with possible phase) equivalent to U up to eps.

        Based on this answer https://shorturl.at/aA079.
        Let x_mask be the index of the maximum number of the first column of U
        and z_mask be the index of the maximum number of the first column of H†UH
        each of these indicies is n-bits long where U is 2^n x 2^n.

        These two indices/masks encode in binary the indices of the qubits that
        have I, X, Y, Z acting on them as follows:
        x_mask[i] == 1 and z_mask[i] == 0: X acts on the ith qubit
        x_mask[i] == 0 and z_mask[i] == 1: Z acts on the ith qubit
        x_mask[i] == 1 and z_mask[i] == 1: Y acts on the ith qubit
        x_mask[i] == 0 and z_mask[i] == 0: I acts on the ith qubit

    Args:
        U: A square array whose dimension is a power of 2.
        eps: numbers smaller than `eps` are considered zero.

    Returns:
        A DensePauliString of None.

    Raises:
        ValueError: if U is not square with a power of 2 dimension.
    """
    if len(U.shape) != 2 or U.shape[0] != U.shape[1]:
        raise ValueError(f'Input has a non-square shape {U}')
    n = U.shape[0].bit_length() - 1
    if U.shape[0] != 2 ** n:
        raise ValueError(f"Input dimension {U.shape[0]} isn't a power of 2")
    x_msk, second_largest = _argmax(U[:, 0])
    if second_largest > eps:
        return None
    U_z = _conjugate_with_hadamard(U)
    z_msk, second_largest = _argmax(U_z[:, 0])
    if second_largest > eps:
        return None

    def select(i):
        """Returns the gate that acts on the ith qubit."""
        has_x = x_msk >> i & 1
        has_z = z_msk >> i & 1
        gate_table = ['IX', 'ZY']
        return gate_table[has_z][has_x]
    decomposition = DensePauliString(''.join((select(i) for i in reversed(range(n)))))
    guess = protocols.unitary(decomposition)
    if np.abs(guess[x_msk, 0]) < eps:
        return None
    phase = U[x_msk, 0] / guess[x_msk, 0]
    phase /= np.abs(phase)
    decomposition = DensePauliString(''.join((select(i) for i in reversed(range(n)))), coefficient=phase)
    if not _validate_decomposition(decomposition, U, eps):
        return None
    return decomposition