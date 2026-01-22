import os
import numpy as np
import pennylane as qml
from pennylane.operation import active_new_opmath
def molecular_hamiltonian(symbols, coordinates, name='molecule', charge=0, mult=1, basis='sto-3g', method='dhf', active_electrons=None, active_orbitals=None, mapping='jordan_wigner', outpath='.', wires=None, alpha=None, coeff=None, args=None, load_data=False, convert_tol=1000000000000.0):
    """Generate the qubit Hamiltonian of a molecule.

    This function drives the construction of the second-quantized electronic Hamiltonian
    of a molecule and its transformation to the basis of Pauli matrices.

    The net charge of the molecule can be given to simulate cationic/anionic systems. Also, the
    spin multiplicity can be input to determine the number of unpaired electrons occupying the HF
    orbitals as illustrated in the left panel of the figure below.

    The basis of Gaussian-type *atomic* orbitals used to represent the *molecular* orbitals can be
    specified to go beyond the minimum basis approximation.

    An active space can be defined for a given number of *active electrons* occupying a reduced set
    of *active orbitals* as sketched in the right panel of the figure below.

    |

    .. figure:: ../../_static/qchem/fig_mult_active_space.png
        :align: center
        :width: 90%

    |

    Args:
        symbols (list[str]): symbols of the atomic species in the molecule
        coordinates (array[float]): atomic positions in Cartesian coordinates.
            The atomic coordinates must be in atomic units and can be given as either a 1D array of
            size ``3*N``, or a 2D array of shape ``(N, 3)`` where ``N`` is the number of atoms.
        name (str): name of the molecule
        charge (int): Net charge of the molecule. If not specified a neutral system is assumed.
        mult (int): Spin multiplicity :math:`\\mathrm{mult}=N_\\mathrm{unpaired} + 1`
            for :math:`N_\\mathrm{unpaired}` unpaired electrons occupying the HF orbitals.
            Possible values of ``mult`` are :math:`1, 2, 3, \\ldots`. If not specified,
            a closed-shell HF state is assumed.
        basis (str): atomic basis set used to represent the molecular orbitals
        method (str): Quantum chemistry method used to solve the
            mean field electronic structure problem. Available options are ``method="dhf"``
            to specify the built-in differentiable Hartree-Fock solver, or ``method="pyscf"``
            to use the OpenFermion-PySCF plugin (this requires ``openfermionpyscf`` to be installed).
        active_electrons (int): Number of active electrons. If not specified, all electrons
            are considered to be active.
        active_orbitals (int): Number of active orbitals. If not specified, all orbitals
            are considered to be active.
        mapping (str): transformation used to map the fermionic Hamiltonian to the qubit Hamiltonian
        outpath (str): path to the directory containing output files
        wires (Wires, list, tuple, dict): Custom wire mapping for connecting to Pennylane ansatz.
            For types ``Wires``/``list``/``tuple``, each item in the iterable represents a wire label
            corresponding to the qubit number equal to its index.
            For type dict, only int-keyed dict (for qubit-to-wire conversion) is accepted for
            partial mapping. If None, will use identity map.
        alpha (array[float]): exponents of the primitive Gaussian functions
        coeff (array[float]): coefficients of the contracted Gaussian functions
        args (array[array[float]]): initial values of the differentiable parameters
        load_data (bool): flag to load data from the basis-set-exchange library
        convert_tol (float): Tolerance in `machine epsilon <https://numpy.org/doc/stable/reference/generated/numpy.real_if_close.html>`_
            for the imaginary part of the Hamiltonian coefficients created by openfermion.
            Coefficients with imaginary part less than 2.22e-16*tol are considered to be real.

    Returns:
        tuple[pennylane.Hamiltonian, int]: the fermionic-to-qubit transformed Hamiltonian
        and the number of qubits

    **Example**

    >>> symbols, coordinates = (['H', 'H'], np.array([0., 0., -0.66140414, 0., 0., 0.66140414]))
    >>> H, qubits = molecular_hamiltonian(symbols, coordinates)
    >>> print(qubits)
    4
    >>> print(H)
    (-0.04207897647782188) [I0]
    + (0.17771287465139934) [Z0]
    + (0.1777128746513993) [Z1]
    + (-0.24274280513140484) [Z2]
    + (-0.24274280513140484) [Z3]
    + (0.17059738328801055) [Z0 Z1]
    + (0.04475014401535161) [Y0 X1 X2 Y3]
    + (-0.04475014401535161) [Y0 Y1 X2 X3]
    + (-0.04475014401535161) [X0 X1 Y2 Y3]
    + (0.04475014401535161) [X0 Y1 Y2 X3]
    + (0.12293305056183801) [Z0 Z2]
    + (0.1676831945771896) [Z0 Z3]
    + (0.1676831945771896) [Z1 Z2]
    + (0.12293305056183801) [Z1 Z3]
    + (0.176276408043196) [Z2 Z3]
    """
    if method not in ['dhf', 'pyscf']:
        raise ValueError("Only 'dhf' and 'pyscf' backends are supported.")
    if len(coordinates) == len(symbols) * 3:
        geometry_dhf = qml.numpy.array(coordinates.reshape(len(symbols), 3))
        geometry_hf = coordinates
    elif len(coordinates) == len(symbols):
        geometry_dhf = qml.numpy.array(coordinates)
        geometry_hf = coordinates.flatten()
    if method == 'dhf':
        if wires:
            wires_new = qml.qchem.convert._process_wires(wires)
            wires_map = dict(zip(range(len(wires_new)), list(wires_new.labels)))
        if mapping != 'jordan_wigner':
            raise ValueError("Only 'jordan_wigner' mapping is supported for the differentiable workflow.")
        if mult != 1:
            raise ValueError("Openshell systems are not supported for the differentiable workflow. Use `method = 'pyscf'` or change the charge or spin multiplicity of the molecule.")
        if args is None and isinstance(geometry_dhf, qml.numpy.tensor):
            geometry_dhf.requires_grad = False
        mol = qml.qchem.Molecule(symbols, geometry_dhf, charge=charge, mult=mult, basis_name=basis, load_data=load_data, alpha=alpha, coeff=coeff)
        core, active = qml.qchem.active_space(mol.n_electrons, mol.n_orbitals, mult, active_electrons, active_orbitals)
        requires_grad = args is not None
        h = qml.qchem.diff_hamiltonian(mol, core=core, active=active)(*args) if requires_grad else qml.qchem.diff_hamiltonian(mol, core=core, active=active)()
        if active_new_opmath():
            h_as_ps = qml.pauli.pauli_sentence(h)
            coeffs = qml.numpy.real(list(h_as_ps.values()), requires_grad=requires_grad)
            h_as_ps = qml.pauli.PauliSentence(dict(zip(h_as_ps.keys(), coeffs)))
            h = qml.s_prod(0, qml.Identity(h.wires[0])) if len(h_as_ps) == 0 else h_as_ps.operation()
        else:
            coeffs = qml.numpy.real(h.coeffs, requires_grad=requires_grad)
            h = qml.Hamiltonian(coeffs, h.ops)
        if wires:
            h = qml.map_wires(h, wires_map)
        return (h, 2 * len(active))
    openfermion, _ = _import_of()
    hf_file = meanfield(symbols, geometry_hf, name, charge, mult, basis, method, outpath)
    molecule = openfermion.MolecularData(filename=hf_file)
    core, active = qml.qchem.active_space(molecule.n_electrons, molecule.n_orbitals, mult, active_electrons, active_orbitals)
    h_of, qubits = (decompose(hf_file, mapping, core, active), 2 * len(active))
    h_pl = qml.qchem.convert.import_operator(h_of, wires=wires, tol=convert_tol)
    return (h_pl, qubits)