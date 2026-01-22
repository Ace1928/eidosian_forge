import itertools as it
import warnings
from functools import partial
from typing import Sequence, Callable
import numpy as np
import pennylane as qml
from pennylane.measurements import ProbabilityMP, StateMP, VarianceMP
from pennylane.transforms import transform
from .general_shift_rules import (
from .gradient_transform import find_and_validate_gradient_methods
from .parameter_shift import _get_operation_recipe
from .hessian_transform import _process_jacs
@partial(transform, classical_cotransform=_contract_qjac_with_cjac, final_transform=True)
def param_shift_hessian(tape: qml.tape.QuantumTape, argnum=None, diagonal_shifts=None, off_diagonal_shifts=None, f0=None) -> (Sequence[qml.tape.QuantumTape], Callable):
    """Transform a circuit to compute the parameter-shift Hessian with respect to its trainable
    parameters. This is the Hessian transform to replace the old one in the new return types system

    Use this transform to explicitly generate and explore parameter-shift circuits for computing
    the Hessian of QNodes directly, without computing first derivatives.

    For second-order derivatives of more complicated cost functions, please consider using your
    chosen autodifferentiation framework directly, by chaining gradient computations:

    >>> qml.jacobian(qml.grad(cost))(weights)

    Args:
        tape (QNode or QuantumTape): quantum circuit to differentiate
        argnum (int or list[int] or array_like[bool] or None): Parameter indices to differentiate
            with respect to. If not provided, the Hessian with respect to all
            trainable indices is returned. Note that the indices refer to tape
            parameters both if ``tape`` is a tape, and if it is a QNode. If an ``array_like``
            is provided, it is expected to be a symmetric two-dimensional Boolean mask with
            shape ``(n, n)`` where ``n`` is the number of trainable tape parameters.
        diagonal_shifts (list[tuple[int or float]]): List containing tuples of shift values
            for the Hessian diagonal. The shifts are understood as first-order derivative
            shifts and are iterated to obtain the second-order derivative.
            If provided, one tuple of shifts should be given per trainable parameter
            and the tuple length should match the number of frequencies for that parameter.
            If unspecified, equidistant shifts are used.
        off_diagonal_shifts (list[tuple[int or float]]): List containing tuples of shift
            values for the off-diagonal entries of the Hessian.
            If provided, one tuple of shifts should be given per trainable parameter
            and the tuple should match the number of frequencies for that parameter.
            The combination of shifts into bivariate shifts is performed automatically.
            If unspecified, equidistant shifts are used.
        f0 (tensor_like[float] or None): Output of the evaluated input tape. If provided,
            and the Hessian tapes include the original input tape, the 'f0' value is used
            instead of evaluating the input tape, reducing the number of device invocations.

    Returns:
        qnode (QNode) or tuple[List[QuantumTape], function]:

        The transformed circuit as described in :func:`qml.transform <pennylane.transform>`. Executing this circuit
        will provide the Hessian in the form of a tensor, a tuple, or a nested tuple depending upon the number
        of trainable QNode arguments, the output shape(s) of the input QNode itself, and the usage of shot vectors
        in the QNode execution.


        Note: By default a QNode with the keyword ``hybrid=True`` computes derivates with respect to
        QNode arguments, which can include classical computations on those arguments before they are
        passed to quantum operations. The "purely quantum" Hessian can instead be obtained with
        ``hybrid=False``, which is then computed with respect to the gate arguments and produces a
        result of shape ``(*QNode output dimensions, # gate arguments, # gate arguments)``.

    **Example**

    Applying the Hessian transform to a QNode computes its Hessian tensor.
    This works best if no classical processing is applied within the
    QNode to operation parameters.

    >>> dev = qml.device("default.qubit", wires=2)
    >>> @qml.qnode(dev)
    ... def circuit(x):
    ...     qml.RX(x[0], wires=0)
    ...     qml.CRY(x[1], wires=[0, 1])
    ...     return qml.expval(qml.Z(0) @ qml.Z(1))

    >>> x = np.array([0.5, 0.2], requires_grad=True)
    >>> qml.gradients.param_shift_hessian(circuit)(x)
    ((array(-0.86883595), array(0.04762358)),
     (array(0.04762358), array(0.05998862)))

    .. details::
        :title: Usage Details

        The Hessian transform can also be applied to a quantum tape instead of a QNode, producing
        the parameter-shifted tapes and a post-processing function to combine the execution
        results of these tapes into the Hessian:

        >>> circuit(x)  # generate the QuantumTape inside the QNode
        >>> tape = circuit.qtape
        >>> hessian_tapes, postproc_fn = qml.gradients.param_shift_hessian(tape)
        >>> len(hessian_tapes)
        13
        >>> all(isinstance(tape, qml.tape.QuantumTape) for tape in hessian_tapes)
        True
        >>> postproc_fn(qml.execute(hessian_tapes, dev, None))
        ((array(-0.86883595), array(0.04762358)),
         (array(0.04762358), array(0.05998862)))

        The Hessian tapes can be inspected via their draw function, which reveals the different
        gate arguments generated from parameter-shift rules (we only draw the first four out of
        all 13 tapes here):

        >>> for h_tape in hessian_tapes[0:4]:
        ...     print(qml.drawer.tape_text(h_tape, decimals=1))
        0: ──RX(0.5)─╭●───────┤ ╭<Z@Z>
        1: ──────────╰RY(0.2)─┤ ╰<Z@Z>
        0: ──RX(-2.6)─╭●───────┤ ╭<Z@Z>
        1: ───────────╰RY(0.2)─┤ ╰<Z@Z>
        0: ──RX(2.1)─╭●───────┤ ╭<Z@Z>
        1: ──────────╰RY(1.8)─┤ ╰<Z@Z>
        0: ──RX(2.1)─╭●────────┤ ╭<Z@Z>
        1: ──────────╰RY(-1.4)─┤ ╰<Z@Z>

        To enable more detailed control over the parameter shifts, shift values can be provided
        per parameter, and separately for the diagonal and the off-diagonal terms.
        Here we choose them based on the parameters ``x`` themselves, mostly yielding multiples of
        the original parameters in the shifted tapes.

        >>> diag_shifts = [(x[0] / 2,), (x[1] / 2, x[1])]
        >>> offdiag_shifts = [(x[0],), (x[1], 2 * x[1])]
        >>> hessian_tapes, postproc_fn = qml.gradients.param_shift_hessian(
        ...     tape, diagonal_shifts=diag_shifts, off_diagonal_shifts=offdiag_shifts
        ... )
        >>> for h_tape in hessian_tapes[0:4]:
        ...     print(qml.drawer.tape_text(h_tape, decimals=1))
        0: ──RX(0.5)─╭●───────┤ ╭<Z@Z>
        1: ──────────╰RY(0.2)─┤ ╰<Z@Z>
        0: ──RX(0.0)─╭●───────┤ ╭<Z@Z>
        1: ──────────╰RY(0.2)─┤ ╰<Z@Z>
        0: ──RX(1.0)─╭●───────┤ ╭<Z@Z>
        1: ──────────╰RY(0.2)─┤ ╰<Z@Z>
        0: ──RX(1.0)─╭●───────┤ ╭<Z@Z>
        1: ──────────╰RY(0.4)─┤ ╰<Z@Z>

        .. note::

            Note that the ``diagonal_shifts`` are interpreted as *first-order* derivative
            shift values. That means they are used to generate a first-order derivative
            recipe, which then is iterated in order to obtain the second-order derivative
            for the diagonal Hessian entry. Explicit control over the used second-order
            shifts is not implemented.

        Finally, the ``argnum`` argument can be used to compute the Hessian only for some of the
        variational parameters. Note that this indexing refers to trainable tape parameters both
        if ``tape`` is a ``QNode`` and if it is a ``QuantumTape``.

        >>> hessian_tapes, postproc_fn = qml.gradients.param_shift_hessian(tape, argnum=(1,))
        >>> postproc_fn(qml.execute(hessian_tapes, dev, None))
        ((array(0.), array(0.)), (array(0.), array(0.05998862)))

    """
    if any((isinstance(m, StateMP) for m in tape.measurements)):
        raise ValueError('Computing the Hessian of circuits that return the state is not supported.')
    if any((isinstance(m, VarianceMP) for m in tape.measurements)):
        raise ValueError('Computing the Hessian of circuits that return variances is currently not supported.')
    if argnum is None and (not tape.trainable_params):
        return _no_trainable_hessian(tape)
    bool_argnum = _process_argnum(argnum, tape)
    compare_diag_to = qml.math.sum(qml.math.diag(bool_argnum))
    offdiag = bool_argnum ^ qml.math.diag(qml.math.diag(bool_argnum))
    compare_offdiag_to = qml.math.sum(qml.math.any(offdiag, axis=0))
    if diagonal_shifts is not None and len(diagonal_shifts) != compare_diag_to:
        raise ValueError(f'The number of provided sets of shift values for diagonal entries ({len(diagonal_shifts)}) does not match the number of marked arguments to compute the diagonal for ({compare_diag_to}).')
    if off_diagonal_shifts is not None and len(off_diagonal_shifts) != compare_offdiag_to:
        raise ValueError(f'The number of provided sets of shift values for off-diagonal entries ({len(off_diagonal_shifts)}) does not match the number of marked arguments for which to compute at least one off-diagonal entry ({compare_offdiag_to}).')
    method = 'analytic' if argnum is None else 'best'
    trainable_params = qml.math.where(qml.math.any(bool_argnum, axis=0))[0]
    diff_methods = find_and_validate_gradient_methods(tape, method, list(trainable_params))
    for i, g in diff_methods.items():
        if g == '0':
            bool_argnum[i] = bool_argnum[:, i] = False
    if qml.math.all(~bool_argnum):
        return _all_zero_hessian(tape)
    unsupported_params = {i for i, m in diff_methods.items() if m == 'F'}
    if unsupported_params:
        raise ValueError(f'The parameter-shift Hessian currently does not support the operations for parameter(s) {unsupported_params}.')
    return expval_hessian_param_shift(tape, bool_argnum, diff_methods, diagonal_shifts, off_diagonal_shifts, f0)