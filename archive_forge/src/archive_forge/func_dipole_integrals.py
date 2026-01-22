import pennylane as qml
from pennylane.fermi import FermiSentence, FermiWord
from .basis_data import atomic_numbers
from .hartree_fock import scf
from .matrices import moment_matrix
from .observable_hf import fermionic_observable, qubit_observable
def dipole_integrals(mol, core=None, active=None):
    """Return a function that computes the dipole moment integrals over the molecular orbitals.

    These integrals are required to construct the dipole operator in the second-quantized form

    .. math::

        \\hat{D} = -\\sum_{pq} d_{pq} [\\hat{c}_{p\\uparrow}^\\dagger \\hat{c}_{q\\uparrow} +
        \\hat{c}_{p\\downarrow}^\\dagger \\hat{c}_{q\\downarrow}] -
        \\hat{D}_\\mathrm{c} + \\hat{D}_\\mathrm{n},

    where the coefficients :math:`d_{pq}` are given by the integral of the position operator
    :math:`\\hat{{\\bf r}}` over molecular orbitals
    :math:`\\phi`

    .. math::

        d_{pq} = \\int \\phi_p^*(r) \\hat{{\\bf r}} \\phi_q(r) dr,

    and :math:`\\hat{c}^{\\dagger}` and :math:`\\hat{c}` are the creation and annihilation operators,
    respectively. The contribution of the core orbitals and nuclei are denoted by
    :math:`\\hat{D}_\\mathrm{c}` and :math:`\\hat{D}_\\mathrm{n}`, respectively.

    The molecular orbitals are represented as a linear combination of atomic orbitals as

    .. math::

        \\phi_i(r) = \\sum_{\\nu}c_{\\nu}^i \\chi_{\\nu}(r).

    Using this equation the dipole moment integral :math:`d_{pq}` can be written as

    .. math::

        d_{pq} = \\sum_{\\mu \\nu} C_{p \\mu} d_{\\mu \\nu} C_{\\nu q},

    where :math:`d_{\\mu \\nu}` is the dipole moment integral over the atomic orbitals and :math:`C`
    is the molecular orbital expansion coefficient matrix. The contribution of the core molecular
    orbitals is computed as

    .. math::

        \\hat{D}_\\mathrm{c} = 2 \\sum_{i=1}^{N_\\mathrm{core}} d_{ii},

    where :math:`N_\\mathrm{core}` is the number of core orbitals.

    Args:
        mol (~qchem.molecule.Molecule): the molecule object
        core (list[int]): indices of the core orbitals
        active (list[int]): indices of the active orbitals

    Returns:
        function: function that computes the dipole moment integrals in the molecular orbital basis

    **Example**

    >>> symbols  = ['H', 'H']
    >>> geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], requires_grad = False)
    >>> alpha = np.array([[3.42525091, 0.62391373, 0.1688554],
    >>>                   [3.42525091, 0.62391373, 0.1688554]], requires_grad=True)
    >>> mol = qml.qchem.Molecule(symbols, geometry, alpha=alpha)
    >>> args = [alpha]
    >>> constants, integrals = dipole_integrals(mol)(*args)
    >>> print(integrals)
    (array([[0., 0.],
            [0., 0.]]),
     array([[0., 0.],
            [0., 0.]]),
     array([[ 0.5      , -0.8270995],
            [-0.8270995,  0.5      ]]))
    """

    def _dipole_integrals(*args):
        """Compute the dipole moment integrals in the molecular orbital basis.

        Args:
            *args (array[array[float]]): initial values of the differentiable parameters

        Returns:
            tuple[array[float]]: tuple containing the core orbital contributions and the dipole
            moment integrals
        """
        _, coeffs, _, _, _ = scf(mol)(*args)
        d_x = qml.math.einsum('qr,rs,st->qt', coeffs.T, moment_matrix(mol.basis_set, 1, 0)(*args), coeffs)
        d_y = qml.math.einsum('qr,rs,st->qt', coeffs.T, moment_matrix(mol.basis_set, 1, 1)(*args), coeffs)
        d_z = qml.math.einsum('qr,rs,st->qt', coeffs.T, moment_matrix(mol.basis_set, 1, 2)(*args), coeffs)
        core_x, core_y, core_z = (qml.math.array([0]), qml.math.array([0]), qml.math.array([0]))
        if core is None and active is None:
            return ((core_x, core_y, core_z), (d_x, d_y, d_z))
        for i in core:
            core_x = core_x + 2 * d_x[i][i]
            core_y = core_y + 2 * d_y[i][i]
            core_z = core_z + 2 * d_z[i][i]
        d_x = d_x[qml.math.ix_(active, active)]
        d_y = d_y[qml.math.ix_(active, active)]
        d_z = d_z[qml.math.ix_(active, active)]
        return ((core_x, core_y, core_z), (d_x, d_y, d_z))
    return _dipole_integrals