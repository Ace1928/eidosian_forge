import functools
import itertools
import numpy as np
import scipy
import pennylane as qml
from pennylane.operation import active_new_opmath
from pennylane.pauli import PauliSentence, PauliWord, pauli_sentence, simplify
from pennylane.pauli.utils import _binary_matrix_from_pws
from pennylane.wires import Wires
def taper_operation(operation, generators, paulixops, paulix_sector, wire_order, op_wires=None, op_gen=None):
    """Transform a gate operation with a Clifford operator and then taper qubits.

    The qubit operator for the generator of the gate operation is computed either internally or can be provided
    manually via the ``op_gen`` argument. If this operator commutes with all the :math:`\\mathbb{Z}_2` symmetries of
    the molecular Hamiltonian, then this operator is transformed using the Clifford operators :math:`U` and
    tapered; otherwise it is discarded. Finally, the tapered generator is exponentiated using :class:`~.pennylane.Exp`
    for building the tapered unitary.

    Args:
        operation (Operation or Callable): qubit operation to be tapered, or a function that applies that operation
        generators (list[Hamiltonian]): generators expressed as PennyLane Hamiltonians
        paulixops (list[Operation]):  list of single-qubit Pauli-X operators
        paulix_sector (list[int]): eigenvalues of the Pauli-X operators
        wire_order (Sequence[Any]): order of the wires in the quantum circuit
        op_wires (Sequence[Any]): wires for the operation in case any of the provided ``operation`` or ``op_gen`` are callables
        op_gen (Hamiltonian or Callable): generator of the operation, or a function that returns it in case it cannot be computed internally.

    Returns:
        list[Operation]: list of operations of type :class:`~.pennylane.Exp` implementing tapered unitary operation

    Raises:
        ValueError: optional argument ``op_wires`` is not provided when the provided operation is a callable
        TypeError: optional argument ``op_gen`` is a callable but does not have ``wires`` as its only keyword argument
        NotImplementedError: generator of the operation cannot be constructed internally
        ValueError: optional argument ``op_gen`` is either not a :class:`~.pennylane.Hamiltonian` or a valid generator of the operation

    **Example**

    >>> symbols, geometry = ['He', 'H'], np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4589]])
    >>> mol = qchem.Molecule(symbols, geometry, charge=1)
    >>> H, n_qubits = qchem.molecular_hamiltonian(symbols, geometry, charge=1)
    >>> generators = qchem.symmetry_generators(H)
    >>> paulixops = qchem.paulix_ops(generators, n_qubits)
    >>> paulix_sector = qchem.optimal_sector(H, generators, mol.n_electrons)
    >>> tap_op = qchem.taper_operation(qml.SingleExcitation, generators, paulixops,
    ...                                paulix_sector, wire_order=H.wires, op_wires=[0, 2])
    >>> tap_op(3.14159)
    [Exp(1.5707949999999993j PauliY), Exp(0j Identity)]

    The obtained tapered operation function can then be used within a :class:`~.pennylane.QNode`:

    >>> dev = qml.device('default.qubit', wires=[0, 1])
    >>> @qml.qnode(dev)
    ... def circuit(params):
    ...     tap_op(params[0])
    ...     return qml.expval(qml.Z(0)@qml.Z(1))
    >>> drawer = qml.draw(circuit, show_all_wires=True)
    >>> print(drawer(params=[3.14159]))
    0: ──Exp(0.00+1.57j Y)─┤ ╭<Z@Z>
    1: ────────────────────┤ ╰<Z@Z>

    .. details::
        :title: Usage Details
        :href: usage-taper-operation

        ``qml.taper_operation`` can also be used with the quantum operations, in which case one does not need to specify ``op_wires`` args:

        >>> qchem.taper_operation(qml.SingleExcitation(3.14159, wires=[0, 2]), generators,
        ...                       paulixops, paulix_sector, wire_order=H.wires)
        [Exp(1.570795j PauliY)]

        Moreover, it can also be used within a :class:`~.pennylane.QNode` directly:

        >>> dev = qml.device('default.qubit', wires=[0, 1])
        >>> @qml.qnode(dev)
        ... def circuit(params):
        ...     qchem.taper_operation(qml.DoubleExcitation(params[0], wires=[0, 1, 2, 3]),
        ...                           generators, paulixops, paulix_sector, H.wires)
        ...     return qml.expval(qml.Z(0)@qml.Z(1))
        >>> drawer = qml.draw(circuit, show_all_wires=True)
        >>> print(drawer(params=[3.14159]))
        0: ─╭Exp(-0.00-0.79j X@Y)─╭Exp(-0.00-0.79j Y@X)─┤ ╭<Z@Z>
        1: ─╰Exp(-0.00-0.79j X@Y)─╰Exp(-0.00-0.79j Y@X)─┤ ╰<Z@Z>

        For more involved gates operations such as the ones constructed from matrices, users would need to provide their generators manually
        via the ``op_gen`` argument. The generator can be passed as a :class:`~.pennylane.Hamiltonian`:

        >>> op_fun = qml.QubitUnitary(np.array([[0.+0.j, 0.+0.j, 0.+0.j, 0.-1.j],
        ...                                     [0.+0.j, 0.+0.j, 0.-1.j, 0.+0.j],
        ...                                     [0.+0.j, 0.-1.j, 0.+0.j, 0.+0.j],
        ...                                     [0.-1.j, 0.+0.j, 0.+0.j, 0.+0.j]]), wires=[0, 2])
        >>> op_gen = qml.Hamiltonian([-0.5 * np.pi],
        ...                          [qml.X(0) @ qml.X(2)])
        >>> qchem.taper_operation(op_fun, generators, paulixops, paulix_sector,
        ...                       wire_order=H.wires, op_gen=op_gen)
        [Exp(1.5707963267948957j PauliX)]

        Alternatively, generators can also be specified as a function which returns :class:`~.pennylane.Hamiltonian` and uses ``wires`` as
        its only required keyword argument:

        >>> op_gen = lambda wires: qml.Hamiltonian(
        ...     [0.25, -0.25],
        ...     [qml.X(wires[0]) @ qml.Y(wires[1]),
        ...      qml.Y(wires[0]) @ qml.X(wires[1])])
        >>> qchem.taper_operation(qml.SingleExcitation, generators, paulixops, paulix_sector,
        ...                       wire_order=H.wires, op_wires=[0, 2], op_gen=op_gen)(3.14159)
        [Exp(1.570795j PauliY)]

    .. details::
        :title: Theory
        :href: theory-taper-operation

        Consider :math:`G` to be the generator of a unitrary :math:`V(\\theta)`, i.e.,

        .. math::

            V(\\theta) = e^{i G \\theta}.

        Then, for :math:`V` to have a non-trivial and compatible tapering with the generators of symmetry
        :math:`\\tau`, we should have :math:`[V, \\tau_i] = 0` for all :math:`\\theta` and :math:`\\tau_i`.
        This would hold only when its generator itself commutes with each :math:`\\tau_i`,

        .. math::

            [V, \\tau_i] = 0 \\iff [G, \\tau_i]\\quad \\forall \\theta, \\tau_i.

        By ensuring this, we can taper the generator :math:`G` using the Clifford operators :math:`U`,
        and exponentiate the transformed generator :math:`G^{\\prime}` to obtain a tapered unitary
        :math:`V^{\\prime}`,

        .. math::

            V^{\\prime} \\equiv e^{i U^{\\dagger} G U \\theta} = e^{i G^{\\prime} \\theta}.
    """
    if active_new_opmath():
        raise qml.QuantumFunctionError('This function is currently not supported with the new operator arithmetic framework. Please de-activate it using `qml.operation.disable_new_opmath()`')
    callable_op = callable(operation)
    operation, op_gen = _build_callables(operation, op_wires=op_wires, op_gen=op_gen)
    op_gen = _build_generator(operation, wire_order, op_gen=op_gen)
    if np.all([[qml.is_commuting(op1, op2) for op1, op2 in itertools.product(generator.ops, op_gen.ops)] for generator in generators]) and (not np.all(np.isclose(op_gen.coeffs, np.zeros_like(op_gen.coeffs), rtol=1e-08))):
        gen_tapered = qml.taper(op_gen, generators, paulixops, paulix_sector)
    else:
        gen_tapered = qml.Hamiltonian([], [])
    gen_tapered = qml.simplify(gen_tapered)

    def _tapered_op(params):
        """Applies the tapered operation for the specified parameter value whenever
        queing context is active, otherwise returns it as a list."""
        if qml.QueuingManager.recording():
            qml.QueuingManager.remove(operation)
            for coeff, op in zip(*gen_tapered.terms()):
                qml.exp(op, 1j * params * coeff)
        else:
            ops_tapered = []
            for coeff, op in zip(*gen_tapered.terms()):
                ops_tapered.append(qml.exp(op, 1j * params * coeff))
            return ops_tapered
    if callable_op:
        return _tapered_op
    params = 1.0
    if operation.parameters and isinstance(operation.parameters[0], (float, complex)):
        params = functools.reduce(lambda i, j: i * j, operation.parameters)
    return _tapered_op(params=params)