import os
import numpy as np
import pennylane as qml
from pennylane.operation import active_new_opmath
def observable(fermion_ops, init_term=0, mapping='jordan_wigner', wires=None):
    """Builds the fermionic many-body observable whose expectation value can be
    measured in PennyLane.

    The second-quantized operator of the fermionic many-body system can combine one-particle
    and two-particle operators as in the case of electronic Hamiltonians :math:`\\hat{H}`:

    .. math::

        \\hat{H} = \\sum_{\\alpha, \\beta} \\langle \\alpha \\vert \\hat{t}^{(1)} +
        \\cdots + \\hat{t}^{(n)} \\vert \\beta \\rangle ~ \\hat{c}_\\alpha^\\dagger \\hat{c}_\\beta
        + \\frac{1}{2} \\sum_{\\alpha, \\beta, \\gamma, \\delta}
        \\langle \\alpha, \\beta \\vert \\hat{v}^{(1)} + \\cdots + \\hat{v}^{(n)}
        \\vert \\gamma, \\delta \\rangle ~ \\hat{c}_\\alpha^\\dagger \\hat{c}_\\beta^\\dagger
        \\hat{c}_\\gamma \\hat{c}_\\delta

    In the latter equations the indices :math:`\\alpha, \\beta, \\gamma, \\delta` run over the
    basis of single-particle states. The operators :math:`\\hat{c}^\\dagger` and :math:`\\hat{c}`
    are the particle creation and annihilation operators, respectively.
    :math:`\\langle \\alpha \\vert \\hat{t} \\vert \\beta \\rangle` denotes the matrix element of
    the single-particle operator :math:`\\hat{t}` entering the observable. For example,
    in electronic structure calculations, this is the case for the kinetic energy operator,
    the nuclei Coulomb potential, or any other external fields included in the Hamiltonian.
    On the other hand, :math:`\\langle \\alpha, \\beta \\vert \\hat{v} \\vert \\gamma, \\delta \\rangle`
    denotes the matrix element of the two-particle operator :math:`\\hat{v}`, for example, the
    Coulomb interaction between the electrons.

    - The observable is built by adding the operators
      :math:`\\sum_{\\alpha, \\beta} t_{\\alpha\\beta}^{(i)}
      \\hat{c}_\\alpha^\\dagger \\hat{c}_\\beta` and
      :math:`\\frac{1}{2} \\sum_{\\alpha, \\beta, \\gamma, \\delta}
      v_{\\alpha\\beta\\gamma\\delta}^{(i)}
      \\hat{c}_\\alpha^\\dagger \\hat{c}_\\beta^\\dagger \\hat{c}_\\gamma \\hat{c}_\\delta`.

    - Second-quantized operators contributing to the
      many-body observable must be represented using the `FermionOperator
      <https://github.com/quantumlib/OpenFermion/blob/master/docs/
      tutorials/intro_to_openfermion.ipynb>`_ data structure as implemented in OpenFermion.
      See the functions :func:`~.one_particle` and :func:`~.two_particle` to build the
      FermionOperator representations of one-particle and two-particle operators.

    - The function uses tools of `OpenFermion <https://github.com/quantumlib/OpenFermion>`_
      to map the resulting fermionic Hamiltonian to the basis of Pauli matrices via the
      Jordan-Wigner or Bravyi-Kitaev transformation. Finally, the qubit operator is converted
      to a PennyLane observable by the function :func:`~.convert_observable`.

    Args:
        fermion_ops (list[FermionOperator]): list containing the FermionOperator data structures
            representing the one-particle and/or two-particle operators entering the many-body
            observable
        init_term (float): Any quantity required to initialize the many-body observable. For
            example, this can be used to pass the nuclear-nuclear repulsion energy :math:`V_{nn}`
            which is typically included in the electronic Hamiltonian of molecules.
        mapping (str): Specifies the fermion-to-qubit mapping. Input values can
            be ``'jordan_wigner'`` or ``'bravyi_kitaev'``.
        wires (Wires, list, tuple, dict): Custom wire mapping used to convert the qubit operator
            to an observable measurable in a PennyLane ansatz.
            For types Wires/list/tuple, each item in the iterable represents a wire label
            corresponding to the qubit number equal to its index.
            For type dict, only int-keyed dict (for qubit-to-wire conversion) is accepted.
            If None, will use identity map (e.g. 0->0, 1->1, ...).

    Returns:
        pennylane.Hamiltonian: the fermionic-to-qubit transformed observable

    **Example**

    >>> t = FermionOperator("0^ 0", 0.5) + FermionOperator("1^ 1", 0.25)
    >>> v = FermionOperator("1^ 0^ 0 1", -0.15) + FermionOperator("2^ 0^ 2 0", 0.3)
    >>> print(observable([t, v], mapping="jordan_wigner"))
    (0.2625) [I0]
    + (-0.1375) [Z0]
    + (-0.0875) [Z1]
    + (-0.0375) [Z0 Z1]
    + (0.075) [Z2]
    + (-0.075) [Z0 Z2]
    """
    openfermion, _ = _import_of()
    if mapping.strip().lower() not in ('jordan_wigner', 'bravyi_kitaev'):
        raise TypeError(f"The '{mapping}' transformation is not available. \n Please set 'mapping' to 'jordan_wigner' or 'bravyi_kitaev'.")
    mb_obs = openfermion.ops.FermionOperator('') * init_term
    for ops in fermion_ops:
        if not isinstance(ops, openfermion.ops.FermionOperator):
            raise TypeError(f"Elements in the lists are expected to be of type 'FermionOperator'; got {type(ops)}")
        mb_obs += ops
    if mapping.strip().lower() == 'bravyi_kitaev':
        return qml.qchem.convert.import_operator(openfermion.transforms.bravyi_kitaev(mb_obs), wires=wires)
    return qml.qchem.convert.import_operator(openfermion.transforms.jordan_wigner(mb_obs), wires=wires)