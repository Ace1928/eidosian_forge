import itertools
import pennylane as qml
from .matrices import core_matrix, mol_density_matrix, overlap_matrix, repulsion_tensor
def hf_energy(mol):
    """Return a function that computes the Hartree-Fock energy.

    Args:
        mol (~qchem.molecule.Molecule): the molecule object

    Returns:
        function: function that computes the Hartree-Fock energy

    **Example**

    >>> symbols  = ['H', 'H']
    >>> geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], requires_grad = False)
    >>> alpha = np.array([[3.42525091, 0.62391373, 0.1688554],
    >>>                   [3.42525091, 0.62391373, 0.1688554]], requires_grad=True)
    >>> mol = qml.qchem.Molecule(symbols, geometry, alpha=alpha)
    >>> args = [alpha]
    >>> hf_energy(mol)(*args)
    -1.065999461545263
    """

    def _hf_energy(*args):
        """Compute the Hartree-Fock energy.

        Args:
            *args (array[array[float]]): initial values of the differentiable parameters

        Returns:
            float: the Hartree-Fock energy
        """
        _, coeffs, fock_matrix, h_core, _ = scf(mol)(*args)
        e_rep = nuclear_energy(mol.nuclear_charges, mol.coordinates)(*args)
        e_elec = qml.math.einsum('pq,qp', fock_matrix + h_core, mol_density_matrix(mol.n_electrons, coeffs))
        energy = e_elec + e_rep
        return energy.reshape(())
    return _hf_energy