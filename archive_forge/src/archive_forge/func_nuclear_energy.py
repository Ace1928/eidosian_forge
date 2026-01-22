import itertools
import pennylane as qml
from .matrices import core_matrix, mol_density_matrix, overlap_matrix, repulsion_tensor
def nuclear_energy(charges, r):
    """Return a function that computes the nuclear-repulsion energy.

    The nuclear-repulsion energy is computed as

    .. math::

        \\sum_{i>j}^n \\frac{q_i q_j}{r_{ij}},

    where :math:`q`, :math:`r` and :math:`n` denote the nuclear charges (atomic numbers), nuclear
    positions and the number of nuclei, respectively.

    Args:
        charges (list[int]): nuclear charges in atomic units
        r (array[float]): nuclear positions

    Returns:
        function: function that computes the nuclear-repulsion energy

    **Example**

    >>> symbols  = ['H', 'F']
    >>> geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]], requires_grad = True)
    >>> mol = qml.qchem.Molecule(symbols, geometry)
    >>> args = [mol.coordinates]
    >>> e = nuclear_energy(mol.nuclear_charges, mol.coordinates)(*args)
    >>> print(e)
    4.5
    """

    def _nuclear_energy(*args):
        """Compute the nuclear-repulsion energy.

        Args:
            *args (array[array[float]]): initial values of the differentiable parameters

        Returns:
            array[float]: nuclear-repulsion energy
        """
        if r.requires_grad:
            coor = args[0]
        else:
            coor = r
        e = qml.math.array([0.0])
        for i, r1 in enumerate(coor):
            for j, r2 in enumerate(coor[i + 1:]):
                e = e + charges[i] * charges[i + j + 1] / qml.math.linalg.norm(r1 - r2)
        return e
    return _nuclear_energy