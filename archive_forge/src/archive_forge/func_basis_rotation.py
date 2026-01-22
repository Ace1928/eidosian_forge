import numpy as np
import pennylane as qml
def basis_rotation(one_electron, two_electron, tol_factor=1e-05):
    """Return the grouped coefficients and observables of a molecular Hamiltonian and the basis
    rotation unitaries obtained with the basis rotation grouping method.

    Args:
        one_electron (array[float]): one-electron integral matrix in the molecular orbital basis
        two_electron (array[array[float]]): two-electron integral tensor in the molecular orbital
            basis arranged in chemist notation
        tol_factor (float): threshold error value for discarding the negligible factors

    Returns:
        tuple(list[array[float]], list[list[Observable]], list[array[float]]): tuple containing
        grouped coefficients, grouped observables and basis rotation transformation matrices

    **Example**

    >>> symbols  = ['H', 'H']
    >>> geometry = np.array([[0.0, 0.0, 0.0],
    ...                      [1.398397361, 0.0, 0.0]], requires_grad=False)
    >>> mol = qml.qchem.Molecule(symbols, geometry)
    >>> core, one, two = qml.qchem.electron_integrals(mol)()
    >>> coeffs, ops, unitaries = basis_rotation(one, two, tol_factor=1.0e-5)
    >>> print(coeffs)
    [array([ 0.84064649, -2.59579282,  0.84064649,  0.45724992,  0.45724992]),
     array([ 9.57150297e-05,  5.60006390e-03,  9.57150297e-05,  2.75092558e-03,
            -9.73801723e-05, -2.79878310e-03, -9.73801723e-05, -2.79878310e-03,
            -2.79878310e-03, -2.79878310e-03,  2.84747318e-03]),
     array([ 0.04530262, -0.04530262, -0.04530262, -0.04530262, -0.04530262,
            0.09060523,  0.04530262]),
     array([-0.66913628,  1.6874169 , -0.66913628,  0.16584151, -0.68077716,
            0.16872663, -0.68077716,  0.16872663,  0.16872663,  0.16872663,
            0.17166195])]

    .. details::
        :title: Theory

        A second-quantized molecular Hamiltonian can be constructed in the
        `chemist notation <http://vergil.chemistry.gatech.edu/notes/permsymm/permsymm.pdf>`_ format
        following Eq. (1) of
        [`PRX Quantum 2, 030305, 2021 <https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.2.030305>`_]
        as

        .. math::

            H = \\sum_{\\alpha \\in \\{\\uparrow, \\downarrow \\} } \\sum_{pq} T_{pq} a_{p,\\alpha}^{\\dagger}
            a_{q, \\alpha} + \\frac{1}{2} \\sum_{\\alpha, \\beta \\in \\{\\uparrow, \\downarrow \\} } \\sum_{pqrs}
            V_{pqrs} a_{p, \\alpha}^{\\dagger} a_{q, \\alpha} a_{r, \\beta}^{\\dagger} a_{s, \\beta},

        where :math:`V_{pqrs}` denotes a two-electron integral in the chemist notation and
        :math:`T_{pq}` is obtained from the one- and two-electron integrals, :math:`h_{pq}` and
        :math:`h_{pqrs}`, as

        .. math::

            T_{pq} = h_{pq} - \\frac{1}{2} \\sum_s h_{pssq}.

        The tensor :math:`V` can be converted to a matrix which is indexed by the indices :math:`pq`
        and :math:`rs` and eigendecomposed up to a rank :math:`R` to give

        .. math::

            V_{pqrs} = \\sum_r^R L_{pq}^{(r)} L_{rs}^{(r) T},

        where :math:`L` denotes the matrix of eigenvectors of the matrix :math:`V`. The molecular
        Hamiltonian can then be rewritten following Eq. (7) of
        [`Phys. Rev. Research 3, 033055, 2021 <https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.3.033055>`_]
        as

        .. math::

            H = \\sum_{\\alpha \\in \\{\\uparrow, \\downarrow \\} } \\sum_{pq} T_{pq} a_{p,\\alpha}^{\\dagger}
            a_{q, \\alpha} + \\frac{1}{2} \\sum_r^R \\left ( \\sum_{\\alpha \\in \\{\\uparrow, \\downarrow \\} } \\sum_{pq}
            L_{pq}^{(r)} a_{p, \\alpha}^{\\dagger} a_{q, \\alpha} \\right )^2.

        The orbital basis can be rotated such that each :math:`T` and :math:`L^{(r)}` matrix is
        diagonal. The Hamiltonian can then be written following Eq. (2) of
        [`npj Quantum Information, 7, 23 (2021) <https://www.nature.com/articles/s41534-020-00341-7>`_]
        as

        .. math::

            H = U_0 \\left ( \\sum_p d_p n_p \\right ) U_0^{\\dagger} + \\sum_r^R U_r \\left ( \\sum_{pq}
            d_{pq}^{(r)} n_p n_q \\right ) U_r^{\\dagger},

        where the coefficients :math:`d` are obtained by diagonalizing the :math:`T` and
        :math:`L^{(r)}` matrices. The number operators :math:`n_p = a_p^{\\dagger} a_p` can be
        converted to qubit operators using

        .. math::

            n_p = \\frac{1-Z_p}{2},

        where :math:`Z_p` is the Pauli :math:`Z` operator applied to qubit :math:`p`. This gives
        the qubit Hamiltonian

        .. math::

           H = U_0 \\left ( \\sum_p O_p^{(0)} \\right ) U_0^{\\dagger} + \\sum_r^R U_r \\left ( \\sum_{q} O_q^{(r)} \\right ) U_r^{\\dagger},

        where :math:`O = \\sum_i c_i P_i` is a linear combination of Pauli words :math:`P_i` that are
        a tensor product of Pauli :math:`Z` and Identity operators. This allows all the Pauli words
        in each of the :math:`O` terms to be measured simultaneously. This function returns the
        coefficients and the Pauli words grouped for each of the :math:`O` terms as well as the
        basis rotation transformation matrices that are constructed from the eigenvectors of the
        :math:`T` and :math:`L^{(r)}` matrices. Each column of the transformation matrix is an
        eigenvector of the corresponding :math:`T` or :math:`L^{(r)}` matrix.
    """
    num_orbitals = one_electron.shape[0] * 2
    one_body_tensor, chemist_two_body_tensor = _chemist_transform(one_electron, two_electron)
    chemist_one_body_tensor = np.kron(one_body_tensor, np.eye(2))
    t_eigvals, t_eigvecs = np.linalg.eigh(chemist_one_body_tensor)
    factors, _, _ = factorize(chemist_two_body_tensor, tol_factor=tol_factor)
    factors = [np.kron(factor, np.eye(2)) for factor in factors]
    v_coeffs, v_unitaries = np.linalg.eigh(factors)
    indices = [np.argsort(v_coeff)[::-1] for v_coeff in v_coeffs]
    v_coeffs = [v_coeff[indices[idx]] for idx, v_coeff in enumerate(v_coeffs)]
    v_unitaries = [v_unitary[:, indices[idx]] for idx, v_unitary in enumerate(v_unitaries)]
    ops_t = 0.0
    for p in range(num_orbitals):
        ops_t += 0.5 * t_eigvals[p] * (qml.Identity(p) - qml.Z(p))
    ops_l = []
    for idx in range(len(factors)):
        ops_l_ = 0.0
        for p in range(num_orbitals):
            for q in range(num_orbitals):
                ops_l_ += v_coeffs[idx][p] * v_coeffs[idx][q] * 0.25 * (qml.Identity(p) - qml.Z(p) - qml.Z(q) + (qml.Identity(p) if p == q else qml.Z(p) @ qml.Z(q)))
        ops_l.append(ops_l_)
    ops = [ops_t] + ops_l
    c_group = [op.coeffs for op in ops]
    o_group = [op.ops for op in ops]
    u_transform = list([t_eigvecs] + list(v_unitaries))
    return (c_group, o_group, u_transform)