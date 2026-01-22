import os
import numpy as np
import pennylane as qml
from pennylane.operation import active_new_opmath
def dipole_of(symbols, coordinates, name='molecule', charge=0, mult=1, basis='sto-3g', package='pyscf', core=None, active=None, mapping='jordan_wigner', cutoff=1e-12, outpath='.', wires=None):
    """Computes the electric dipole moment operator in the Pauli basis.

    The second quantized dipole moment operator :math:`\\hat{D}` of a molecule is given by

    .. math::

        \\hat{D} = -\\sum_{\\alpha, \\beta} \\langle \\alpha \\vert \\hat{{\\bf r}} \\vert \\beta \\rangle
        [\\hat{c}_{\\alpha\\uparrow}^\\dagger \\hat{c}_{\\beta\\uparrow} +
        \\hat{c}_{\\alpha\\downarrow}^\\dagger \\hat{c}_{\\beta\\downarrow}] + \\hat{D}_\\mathrm{n}.

    In the equation above, the indices :math:`\\alpha, \\beta` run over the basis of Hartree-Fock
    molecular orbitals and the operators :math:`\\hat{c}^\\dagger` and :math:`\\hat{c}` are the
    electron creation and annihilation operators, respectively. The matrix elements of the
    position operator :math:`\\hat{{\\bf r}}` are computed as

    .. math::

        \\langle \\alpha \\vert \\hat{{\\bf r}} \\vert \\beta \\rangle = \\sum_{i, j}
         C_{\\alpha i}^*C_{\\beta j} \\langle i \\vert \\hat{{\\bf r}} \\vert j \\rangle,

    where :math:`\\vert i \\rangle` is the wave function of the atomic orbital,
    :math:`C_{\\alpha i}` are the coefficients defining the molecular orbitals,
    and :math:`\\langle i \\vert \\hat{{\\bf r}} \\vert j \\rangle`
    is the representation of operator :math:`\\hat{{\\bf r}}` in the atomic basis.

    The contribution of the nuclei to the dipole operator is given by

    .. math::

        \\hat{D}_\\mathrm{n} = \\sum_{i=1}^{N_\\mathrm{atoms}} Z_i {\\bf R}_i \\hat{I},


    where :math:`Z_i` and :math:`{\\bf R}_i` denote, respectively, the atomic number and the
    nuclear coordinates of the :math:`i`-th atom of the molecule.

    Args:
        symbols (list[str]): symbols of the atomic species in the molecule
        coordinates (array[float]): 1D array with the atomic positions in Cartesian
            coordinates. The coordinates must be given in atomic units and the size of the array
            should be ``3*N`` where ``N`` is the number of atoms.
        name (str): name of the molecule
        charge (int): charge of the molecule
        mult (int): spin multiplicity :math:`\\mathrm{mult}=N_\\mathrm{unpaired} + 1` of the
            Hartree-Fock (HF) state based on the number of unpaired electrons occupying the
            HF orbitals
        basis (str): Atomic basis set used to represent the molecular orbitals. Basis set
            availability per element can be found
            `here <www.psicode.org/psi4manual/master/basissets_byelement.html#apdx-basiselement>`_
        package (str): quantum chemistry package (pyscf) used to solve the
            mean field electronic structure problem
        core (list): indices of core orbitals
        active (list): indices of active orbitals
        mapping (str): transformation (``'jordan_wigner'`` or ``'bravyi_kitaev'``) used to
            map the fermionic operator to the Pauli basis
        cutoff (float): Cutoff value for including the matrix elements
            :math:`\\langle \\alpha \\vert \\hat{{\\bf r}} \\vert \\beta \\rangle`. The matrix elements
            with absolute value less than ``cutoff`` are neglected.
        outpath (str): path to the directory containing output files
        wires (Wires, list, tuple, dict): Custom wire mapping used to convert the qubit operator
            to an observable measurable in a PennyLane ansatz.
            For types Wires/list/tuple, each item in the iterable represents a wire label
            corresponding to the qubit number equal to its index.
            For type dict, only int-keyed dict (for qubit-to-wire conversion) is accepted.
            If None, will use identity map (e.g. 0->0, 1->1, ...).

    Returns:
        list[pennylane.Hamiltonian]: the qubit observables corresponding to the components
        :math:`\\hat{D}_x`, :math:`\\hat{D}_y` and :math:`\\hat{D}_z` of the dipole operator in
        atomic units.

    **Example**

    >>> symbols = ["H", "H", "H"]
    >>> coordinates = np.array([0.028, 0.054, 0.0, 0.986, 1.610, 0.0, 1.855, 0.002, 0.0])
    >>> dipole_obs = dipole_of(symbols, coordinates, charge=1)
    >>> print(dipole_obs)
    [<Hamiltonian: terms=18, wires=[0, 1, 2, 3, 4, 5]>,
    <Hamiltonian: terms=18, wires=[0, 1, 2, 3, 4, 5]>,
    <Hamiltonian: terms=1, wires=[0]>]

    >>> print(dipole_obs[0]) # x-component of D
    (0.24190977644628117) [Z4]
    + (0.24190977644628117) [Z5]
    + (0.4781123173263878) [Z0]
    + (0.4781123173263878) [Z1]
    + (0.714477906181248) [Z2]
    + (0.714477906181248) [Z3]
    + (-0.3913638489487808) [Y0 Z1 Y2]
    + (-0.3913638489487808) [X0 Z1 X2]
    + (-0.3913638489487808) [Y1 Z2 Y3]
    + (-0.3913638489487808) [X1 Z2 X3]
    + (-0.1173495878099553) [Y2 Z3 Y4]
    + (-0.1173495878099553) [X2 Z3 X4]
    + (-0.1173495878099553) [Y3 Z4 Y5]
    + (-0.1173495878099553) [X3 Z4 X5]
    + (0.26611147045300276) [Y0 Z1 Z2 Z3 Y4]
    + (0.26611147045300276) [X0 Z1 Z2 Z3 X4]
    + (0.26611147045300276) [Y1 Z2 Z3 Z4 Y5]
    + (0.26611147045300276) [X1 Z2 Z3 Z4 X5]
    """
    openfermion, _ = _import_of()
    atomic_numbers = {'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10}
    if mult != 1:
        raise ValueError(f'Currently, this functionality is constrained to Hartree-Fock states with spin multiplicity = 1; got multiplicity 2S+1 =  {mult}')
    for i in symbols:
        if i not in atomic_numbers:
            raise ValueError(f'Currently, only first- or second-row elements of the periodic table are supported; got element {i}')
    hf_file = qml.qchem.meanfield(symbols, coordinates, name, charge, mult, basis, package, outpath)
    hf = openfermion.MolecularData(filename=hf_file.strip())
    from pyscf import gto
    mol = gto.M(atom=hf.geometry, basis=hf.basis, charge=hf.charge, spin=0.5 * (hf.multiplicity - 1))
    dip_ao = mol.intor_symmetric('int1e_r', comp=3).real
    n_orbs = hf.n_orbitals
    c_hf = hf.canonical_orbitals
    dip_mo = np.zeros((3, n_orbs, n_orbs))
    for comp in range(3):
        for alpha in range(n_orbs):
            for beta in range(alpha + 1):
                dip_mo[comp, alpha, beta] = c_hf[:, alpha] @ dip_ao[comp] @ c_hf[:, beta]
        dip_mo[comp] += dip_mo[comp].T - np.diag(np.diag(dip_mo[comp]))
    dip_n = np.zeros(3)
    for comp in range(3):
        for i, symb in enumerate(symbols):
            dip_n[comp] += atomic_numbers[symb] * coordinates[3 * i + comp]
    dip = []
    for i in range(3):
        fermion_obs = one_particle(dip_mo[i], core=core, active=active, cutoff=cutoff)
        dip.append(observable([-fermion_obs], init_term=dip_n[i], mapping=mapping, wires=wires))
    return dip