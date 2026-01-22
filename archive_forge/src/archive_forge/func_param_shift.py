from typing import Sequence, Callable
from functools import partial
import numpy as np
import pennylane as qml
from pennylane.measurements import VarianceMP
from pennylane import transform
from pennylane.transforms.tape_expand import expand_invalid_trainable
from pennylane.gradients.gradient_transform import _contract_qjac_with_cjac
from .finite_difference import finite_diff
from .general_shift_rules import (
from .gradient_transform import (
@partial(transform, expand_transform=_expand_transform_param_shift, classical_cotransform=_contract_qjac_with_cjac, final_transform=True)
def param_shift(tape: qml.tape.QuantumTape, argnum=None, shifts=None, gradient_recipes=None, fallback_fn=finite_diff, f0=None, broadcast=False) -> (Sequence[qml.tape.QuantumTape], Callable):
    """Transform a circuit to compute the parameter-shift gradient of all gate
    parameters with respect to its inputs.

    Args:
        tape (QNode or QuantumTape): quantum circuit to differentiate
        argnum (int or list[int] or None): Trainable parameter indices to differentiate
            with respect to. If not provided, the derivative with respect to all
            trainable indices are returned.
        shifts (list[tuple[int or float]]): List containing tuples of shift values.
            If provided, one tuple of shifts should be given per trainable parameter
            and the tuple should match the number of frequencies for that parameter.
            If unspecified, equidistant shifts are assumed.
        gradient_recipes (tuple(list[list[float]] or None)): List of gradient recipes
            for the parameter-shift method. One gradient recipe must be provided
            per trainable parameter.

            This is a tuple with one nested list per parameter. For
            parameter :math:`\\phi_k`, the nested list contains elements of the form
            :math:`[c_i, a_i, s_i]` where :math:`i` is the index of the
            term, resulting in a gradient recipe of

            .. math:: \\frac{\\partial}{\\partial\\phi_k}f = \\sum_{i} c_i f(a_i \\phi_k + s_i).

            If ``None``, the default gradient recipe containing the two terms
            :math:`[c_0, a_0, s_0]=[1/2, 1, \\pi/2]` and :math:`[c_1, a_1,
            s_1]=[-1/2, 1, -\\pi/2]` is assumed for every parameter.
        fallback_fn (None or Callable): a fallback gradient function to use for
            any parameters that do not support the parameter-shift rule.
        f0 (tensor_like[float] or None): Output of the evaluated input tape. If provided,
            and the gradient recipe contains an unshifted term, this value is used,
            saving a quantum evaluation.
        broadcast (bool): Whether or not to use parameter broadcasting to create the
            a single broadcasted tape per operation instead of one tape per shift angle.

    Returns:
        qnode (QNode) or tuple[List[QuantumTape], function]:

        The transformed circuit as described in :func:`qml.transform <pennylane.transform>`. Executing this circuit
        will provide the Jacobian in the form of a tensor, a tuple, or a nested tuple depending upon the nesting
        structure of measurements in the original circuit.

    For a variational evolution :math:`U(\\mathbf{p}) \\vert 0\\rangle` with
    :math:`N` parameters :math:`\\mathbf{p}`,
    consider the expectation value of an observable :math:`O`:

    .. math::

        f(\\mathbf{p})  = \\langle \\hat{O} \\rangle(\\mathbf{p}) = \\langle 0 \\vert
        U(\\mathbf{p})^\\dagger \\hat{O} U(\\mathbf{p}) \\vert 0\\rangle.


    The gradient of this expectation value can be calculated via the parameter-shift rule:

    .. math::

        \\frac{\\partial f}{\\partial \\mathbf{p}} = \\sum_{\\mu=1}^{2R}
        f\\left(\\mathbf{p}+\\frac{2\\mu-1}{2R}\\pi\\right)
        \\frac{(-1)^{\\mu-1}}{4R\\sin^2\\left(\\frac{2\\mu-1}{4R}\\pi\\right)}

    Here, :math:`R` is the number of frequencies with which the parameter :math:`\\mathbf{p}`
    enters the function :math:`f` via the operation :math:`U`, and we assumed that these
    frequencies are equidistant.
    For more general shift rules, both regarding the shifts and the frequencies, and
    for more technical details, see
    `Vidal and Theis (2018) <https://arxiv.org/abs/1812.06323>`_ and
    `Wierichs et al. (2022) <https://doi.org/10.22331/q-2022-03-30-677>`_.

    **Gradients of variances**

    For a variational evolution :math:`U(\\mathbf{p}) \\vert 0\\rangle` with
    :math:`N` parameters :math:`\\mathbf{p}`,
    consider the variance of an observable :math:`O`:

    .. math::

        g(\\mathbf{p})=\\langle \\hat{O}^2 \\rangle (\\mathbf{p}) - [\\langle \\hat{O}
        \\rangle(\\mathbf{p})]^2.

    We can relate this directly to the parameter-shift rule by noting that

    .. math::

        \\frac{\\partial g}{\\partial \\mathbf{p}}= \\frac{\\partial}{\\partial
        \\mathbf{p}} \\langle \\hat{O}^2 \\rangle (\\mathbf{p})
        - 2 f(\\mathbf{p}) \\frac{\\partial f}{\\partial \\mathbf{p}}.

    The derivatives in the expression on the right hand side can be computed via
    the shift rule as above, allowing for the computation of the variance derivative.

    In the case where :math:`O` is involutory (:math:`\\hat{O}^2 = I`), the first
    term in the above expression vanishes, and we are simply left with

    .. math::

      \\frac{\\partial g}{\\partial \\mathbf{p}} = - 2 f(\\mathbf{p})
      \\frac{\\partial f}{\\partial \\mathbf{p}}.

    **Example**

    This transform can be registered directly as the quantum gradient transform
    to use during autodifferentiation:

    >>> dev = qml.device("default.qubit", wires=2)
    >>> @qml.qnode(dev, interface="autograd", diff_method="parameter-shift")
    ... def circuit(params):
    ...     qml.RX(params[0], wires=0)
    ...     qml.RY(params[1], wires=0)
    ...     qml.RX(params[2], wires=0)
    ...     return qml.expval(qml.Z(0))
    >>> params = np.array([0.1, 0.2, 0.3], requires_grad=True)
    >>> qml.jacobian(circuit)(params)
    array([-0.3875172 , -0.18884787, -0.38355704])

    When differentiating QNodes with multiple measurements using Autograd or TensorFlow, the outputs of the QNode first
    need to be stacked. The reason is that those two frameworks only allow differentiating functions with array or
    tensor outputs, instead of functions that output sequences. In contrast, Jax and Torch require no additional
    post-processing.

    >>> import jax
    >>> dev = qml.device("default.qubit", wires=2)
    >>> @qml.qnode(dev, interface="jax", diff_method="parameter-shift")
    ... def circuit(params):
    ...     qml.RX(params[0], wires=0)
    ...     qml.RY(params[1], wires=0)
    ...     qml.RX(params[2], wires=0)
    ...     return qml.expval(qml.Z(0)), qml.var(qml.Z(0))
    >>> params = jax.numpy.array([0.1, 0.2, 0.3])
    >>> jax.jacobian(circuit)(params)
    (Array([-0.38751727, -0.18884793, -0.3835571 ], dtype=float32), Array([0.6991687 , 0.34072432, 0.6920237 ], dtype=float32))

    .. note::

        ``param_shift`` performs multiple attempts to obtain the gradient recipes for
        each operation:

        - If an operation has a custom :attr:`~.operation.Operation.grad_recipe` defined,
          it is used.

        - If :attr:`~.operation.Operation.parameter_frequencies` yields a result, the frequencies
          are used to construct the general parameter-shift rule via
          :func:`.generate_shift_rule`.
          Note that by default, the generator is used to compute the parameter frequencies
          if they are not provided via a custom implementation.

        That is, the order of precedence is :attr:`~.operation.Operation.grad_recipe`, custom
        :attr:`~.operation.Operation.parameter_frequencies`, and finally
        :meth:`~.operation.Operation.generator` via the default implementation of the frequencies.

    .. warning::

        Note that using parameter broadcasting via ``broadcast=True`` is not supported for tapes
        with multiple return values or for evaluations with shot vectors.
        As the option ``broadcast=True`` adds a broadcasting dimension, it is not compatible
        with circuits that already are broadcasted.
        Finally, operations with trainable parameters are required to support broadcasting.
        One way of checking this is the `Attribute` `supports_broadcasting`:

        >>> qml.RX in qml.ops.qubit.attributes.supports_broadcasting
        True

    .. details::
        :title: Usage Details

        This gradient transform can be applied directly to :class:`QNode <pennylane.QNode>` objects.
        However, for performance reasons, we recommend providing the gradient transform as the ``diff_method`` argument
        of the QNode decorator, and differentiating with your preferred machine learning framework.

        >>> @qml.qnode(dev)
        ... def circuit(params):
        ...     qml.RX(params[0], wires=0)
        ...     qml.RY(params[1], wires=0)
        ...     qml.RX(params[2], wires=0)
        ...     return qml.expval(qml.Z(0)), qml.var(qml.Z(0))
        >>> qml.gradients.param_shift(circuit)(params)
        ((tensor(-0.38751724, requires_grad=True),
          tensor(-0.18884792, requires_grad=True),
          tensor(-0.38355709, requires_grad=True)),
         (array(0.69916862), array(0.34072424), array(0.69202359)))

        This quantum gradient transform can also be applied to low-level
        :class:`~.QuantumTape` objects. This will result in no implicit quantum
        device evaluation. Instead, the processed tapes, and post-processing
        function, which together define the gradient are directly returned:

        >>> ops = [qml.RX(params[0], 0), qml.RY(params[1], 0), qml.RX(params[2], 0)]
        >>> measurements = [qml.expval(qml.Z(0)), qml.var(qml.Z(0))]
        >>> tape = qml.tape.QuantumTape(ops, measurements)
        >>> gradient_tapes, fn = qml.gradients.param_shift(tape)
        >>> gradient_tapes
        [<QuantumTape: wires=[0, 1], params=3>,
         <QuantumTape: wires=[0, 1], params=3>,
         <QuantumTape: wires=[0, 1], params=3>,
         <QuantumTape: wires=[0, 1], params=3>,
         <QuantumTape: wires=[0, 1], params=3>,
         <QuantumTape: wires=[0, 1], params=3>]

        This can be useful if the underlying circuits representing the gradient
        computation need to be analyzed.

        Note that ``argnum`` refers to the index of a parameter within the list of trainable
        parameters. For example, if we have:

        >>> tape = qml.tape.QuantumScript(
        ...     [qml.RX(1.2, wires=0), qml.RY(2.3, wires=0), qml.RZ(3.4, wires=0)],
        ...     [qml.expval(qml.Z(0))],
        ...     trainable_params = [1, 2]
        ... )
        >>> qml.gradients.param_shift(tape, argnum=1)

        The code above will differentiate the third parameter rather than the second.

        The output tapes can then be evaluated and post-processed to retrieve
        the gradient:

        >>> dev = qml.device("default.qubit", wires=2)
        >>> fn(qml.execute(gradient_tapes, dev, None))
        ((tensor(-0.3875172, requires_grad=True),
          tensor(-0.18884787, requires_grad=True),
          tensor(-0.38355704, requires_grad=True)),
         (array(0.69916862), array(0.34072424), array(0.69202359)))

        This gradient transform is compatible with devices that use shot vectors for execution.

        >>> shots = (10, 100, 1000)
        >>> dev = qml.device("default.qubit", wires=2, shots=shots)
        >>> @qml.qnode(dev)
        ... def circuit(params):
        ...     qml.RX(params[0], wires=0)
        ...     qml.RY(params[1], wires=0)
        ...     qml.RX(params[2], wires=0)
        ...     return qml.expval(qml.Z(0)), qml.var(qml.Z(0))
        >>> params = np.array([0.1, 0.2, 0.3], requires_grad=True)
        >>> qml.gradients.param_shift(circuit)(params)
        (((array(-0.6), array(-0.1), array(-0.1)),
          (array(1.2), array(0.2), array(0.2))),
         ((array(-0.39), array(-0.24), array(-0.49)),
          (array(0.7488), array(0.4608), array(0.9408))),
         ((array(-0.36), array(-0.191), array(-0.37)),
          (array(0.65808), array(0.349148), array(0.67636))))

        The outermost tuple contains results corresponding to each element of the shot vector.

        When setting the keyword argument ``broadcast`` to ``True``, the shifted
        circuit evaluations for each operation are batched together, resulting in
        broadcasted tapes:

        >>> params = np.array([0.1, 0.2, 0.3], requires_grad=True)
        >>> ops = [qml.RX(p, wires=0) for p in params]
        >>> measurements = [qml.expval(qml.Z(0))]
        >>> tape = qml.tape.QuantumTape(ops, measurements)
        >>> gradient_tapes, fn = qml.gradients.param_shift(tape, broadcast=True)
        >>> len(gradient_tapes)
        3
        >>> [t.batch_size for t in gradient_tapes]
        [2, 2, 2]

        The postprocessing function will know that broadcasting is used and handle the results accordingly:

        >>> fn(qml.execute(gradient_tapes, dev, None))
        (array(-0.3875172), array(-0.18884787), array(-0.38355704))

        An advantage of using ``broadcast=True`` is a speedup:

        >>> @qml.qnode(dev)
        ... def circuit(params):
        ...     qml.RX(params[0], wires=0)
        ...     qml.RY(params[1], wires=0)
        ...     qml.RX(params[2], wires=0)
        ...     return qml.expval(qml.Z(0))
        >>> number = 100
        >>> serial_call = "qml.gradients.param_shift(circuit, broadcast=False)(params)"
        >>> timeit.timeit(serial_call, globals=globals(), number=number) / number
        0.020183045039993887
        >>> broadcasted_call = "qml.gradients.param_shift(circuit, broadcast=True)(params)"
        >>> timeit.timeit(broadcasted_call, globals=globals(), number=number) / number
        0.01244492811998498

        This speedup grows with the number of shifts and qubits until all preprocessing and
        postprocessing overhead becomes negligible. While it will depend strongly on the details
        of the circuit, at least a small improvement can be expected in most cases.
        Note that ``broadcast=True`` requires additional memory by a factor of the largest
        batch_size of the created tapes.
    """
    transform_name = 'parameter-shift rule'
    assert_no_state_returns(tape.measurements, transform_name)
    assert_multimeasure_not_broadcasted(tape.measurements, broadcast)
    assert_no_tape_batching(tape, transform_name)
    if argnum is None and (not tape.trainable_params):
        return _no_trainable_grad(tape)
    method = 'analytic' if fallback_fn is None else 'best'
    trainable_params = choose_trainable_params(tape, argnum)
    diff_methods = find_and_validate_gradient_methods(tape, method, trainable_params)
    if all((g == '0' for g in diff_methods.values())):
        return _all_zero_grad(tape)
    unsupported_params = {idx for idx, g in diff_methods.items() if g == 'F'}
    argnum = [i for i, dm in diff_methods.items() if dm == 'A']
    gradient_tapes = []
    if unsupported_params:
        if not argnum:
            return fallback_fn(tape)
        g_tapes, fallback_proc_fn = fallback_fn(tape, argnum=unsupported_params)
        gradient_tapes.extend(g_tapes)
        fallback_len = len(g_tapes)
    if gradient_recipes is None:
        gradient_recipes = [None] * len(argnum)
    if any((isinstance(m, VarianceMP) for m in tape.measurements)):
        g_tapes, fn = var_param_shift(tape, argnum, shifts, gradient_recipes, f0, broadcast)
    else:
        g_tapes, fn = expval_param_shift(tape, argnum, shifts, gradient_recipes, f0, broadcast)
    gradient_tapes.extend(g_tapes)
    if unsupported_params:

        def _single_shot_batch_grad(unsupported_grads, supported_grads):
            """Auxiliary function for post-processing one batch of supported and unsupported gradients corresponding to
            finite shot execution.

            If the device used a shot vector, gradients corresponding to a single component of the shot vector should be
            passed to this aux function.
            """
            multi_measure = len(tape.measurements) > 1
            if not multi_measure:
                res = []
                for i, j in zip(unsupported_grads, supported_grads):
                    component = qml.math.array(i + j)
                    res.append(component)
                return tuple(res)
            combined_grad = []
            for meas_res1, meas_res2 in zip(unsupported_grads, supported_grads):
                meas_grad = []
                for param_res1, param_res2 in zip(meas_res1, meas_res2):
                    component = qml.math.array(param_res1 + param_res2)
                    meas_grad.append(component)
                meas_grad = tuple(meas_grad)
                combined_grad.append(meas_grad)
            return tuple(combined_grad)

        def processing_fn(results):
            unsupported_res = results[:fallback_len]
            supported_res = results[fallback_len:]
            if not tape.shots.has_partitioned_shots:
                unsupported_grads = fallback_proc_fn(unsupported_res)
                supported_grads = fn(supported_res)
                return _single_shot_batch_grad(unsupported_grads, supported_grads)
            supported_grads = fn(supported_res)
            unsupported_grads = fallback_proc_fn(unsupported_res)
            final_grad = []
            for idx in range(tape.shots.num_copies):
                u_grads = unsupported_grads[idx]
                sup = supported_grads[idx]
                final_grad.append(_single_shot_batch_grad(u_grads, sup))
            return tuple(final_grad)
        return (gradient_tapes, processing_fn)
    return (gradient_tapes, fn)