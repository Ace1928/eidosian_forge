import functools
import itertools
import numpy as np
import scipy
import pennylane as qml
from pennylane.operation import active_new_opmath
from pennylane.pauli import PauliSentence, PauliWord, pauli_sentence, simplify
from pennylane.pauli.utils import _binary_matrix_from_pws
from pennylane.wires import Wires
def optimal_sector(qubit_op, generators, active_electrons):
    """Get the optimal sector which contains the ground state.

    To obtain the optimal sector, we need to choose the right eigenvalues for the symmetry generators :math:`\\bm{\\tau}`.
    We can do so by using the following relation between the Pauli-Z qubit operator and the occupation number under a
    Jordan-Wigner transform.

    .. math::

        \\sigma_{i}^{z} = I - 2a_{i}^{\\dagger}a_{i}

    According to this relation, the occupied and unoccupied fermionic modes correspond to the -1 and +1 eigenvalues of
    the Pauli-Z operator, respectively. Since all of the generators :math:`\\bm{\\tau}` consist only of :math:`I` and
    Pauli-Z operators, the correct eigenvalue for each :math:`\\tau` operator can be simply obtained by applying it on
    the reference Hartree-Fock (HF) state, and looking at the overlap between the wires on which the Pauli-Z operators
    act and the wires that correspond to occupied orbitals in the HF state.

    Args:
        qubit_op (Operator): Hamiltonian for which symmetries are being generated
        generators (list[Operator]): list of symmetry generators for the Hamiltonian
        active_electrons (int): The number of active electrons in the system

    Returns:
        list[int]: eigenvalues corresponding to the optimal sector which contains the ground state

    **Example**

    >>> symbols = ["H", "H"]
    >>> geometry = np.array([[0.0, 0.0, -0.69440367], [0.0, 0.0, 0.69440367]])
    >>> H, qubits = qml.qchem.molecular_hamiltonian(symbols, geometry)
    >>> generators = qml.qchem.symmetry_generators(H)
    >>> qml.qchem.optimal_sector(H, generators, 2)
        [1, -1, -1]
    """
    if active_electrons < 1:
        raise ValueError(f"The number of active electrons must be greater than zero;got 'electrons'={active_electrons}")
    num_orbitals = len(qubit_op.wires)
    if active_electrons > num_orbitals:
        raise ValueError(f"Number of active orbitals cannot be smaller than number of active electrons; got 'orbitals'={num_orbitals} < 'electrons'={active_electrons}.")
    hf_str = np.where(np.arange(num_orbitals) < active_electrons, 1, 0)
    perm = []
    for tau in generators:
        symmstr = np.array([1 if wire in tau.wires else 0 for wire in qubit_op.wires.toset()])
        coeff = -1 if np.logical_xor.reduce(np.logical_and(symmstr, hf_str)) else 1
        perm.append(coeff)
    return perm