from copy import deepcopy
import functools
import inspect
import os
import warnings
import pennylane as qml
from pennylane.tape import make_qscript
def qfunc_transform(tape_transform):
    """Given a function which defines a tape transform, convert the function into
    one that applies the tape transform to quantum functions (qfuncs).

    .. warning::

        Use of ``qfunc_transform`` to create a custom transform is deprecated. Instead
        switch to using the new :func:`transform` function. Follow the instructions
        `here <https://docs.pennylane.ai/en/stable/code/qml_transforms.html#custom-transforms>`_
        for further details

    Args:
        tape_transform (function or single_tape_transform): the single tape transform
            to convert into the qfunc transform.

    Returns:
        function: A qfunc transform, that acts on any qfunc, and returns a *new*
        qfunc as per the tape transform. Note that if ``tape_transform`` takes
        additional parameters beyond a single tape, then the created qfunc transform
        will take the *same* parameters, prior to being applied to the qfunc.

    **Example**

    Given a single tape transform ``my_transform(tape, x, y)``, you can use
    this function to convert it into a qfunc transform:

    >>> my_qfunc_transform = qfunc_transform(my_transform)

    It can then be used to transform an existing qfunc:

    >>> new_qfunc = my_qfunc_transform(0.6, 0.7)(old_qfunc)
    >>> new_qfunc(params)

    It can also be used as a decorator:

    .. code-block:: python

        @qml.qfunc_transform
        def my_transform(tape, x, y):
            for op in tape:
                if op.name == "CRX":
                    wires = op.wires
                    param = op.parameters[0]
                    qml.RX(x * param, wires=wires[1])
                    qml.RY(y * qml.math.sqrt(param), wires=wires[1])
                    qml.CZ(wires=[wires[1], wires[0]])
                else:
                    op.queue()

        @my_transform(0.6, 0.1)
        def qfunc(x):
            qml.Hadamard(wires=0)
            qml.CRX(x, wires=[0, 1])
            return qml.expval(qml.Z(1))

    >>> dev = qml.device("default.qubit", wires=2)
    >>> qnode = qml.QNode(qfunc, dev)
    >>> print(qml.draw(qnode)(2.5))
    0: ──H──────────────────╭Z─┤
    1: ──RX(1.50)──RY(0.16)─╰●─┤  <Z>

    The transform weights provided to a qfunc transform are fully differentiable,
    allowing the transform itself to be differentiated and trained. For more details,
    see the Differentiability section under Usage Details.

    .. details::
        :title: Usage Details

        **Inline usage**

        qfunc transforms, when used inline (that is, not as a decorator), take the following form:

        >>> my_transform(transform_weights)(ansatz)(param)

        or

        >>> my_transform(ansatz)(param)

        if they do not permit any parameters. We can break this down into distinct steps,
        to show what is happening with each new function call:

        0. Create a transform defined by the transform weights:

           >>> specific_transform = my_transform(transform_weights)

           Note that this step is skipped if the transform does not provide any
           weights/parameters that can be modified!

        1. Apply the transform to the qfunc. A qfunc transform always acts on
           a qfunc, returning a new qfunc:

           >>> new_qfunc = specific_transform(ansatz)

        2. Finally, we evaluate the new, transformed, qfunc:

           >>> new_qfunc(params)

        So the syntax

        >>> my_transform(transform_weights)(ansatz)(param)

        simply 'chains' these three steps together, into a single call.

        **Differentiability**

        When applying a qfunc transform, not only is the newly transformed qfunc fully
        differentiable, but the qfunc transform parameters *themselves* are differentiable.
        This allows us to train both the quantum function, as well as the transform
        that created it.

        Consider the following example, where a pre-defined ansatz is transformed
        within a QNode:

        .. code-block:: python

            dev = qml.device("default.qubit", wires=2)

            def ansatz(x):
                qml.Hadamard(wires=0)
                qml.CRX(x, wires=[0, 1])

            @qml.qnode(dev)
            def circuit(param, transform_weights):
                qml.RX(0.1, wires=0)

                # apply the transform to the ansatz
                my_transform(*transform_weights)(ansatz)(param)

                return qml.expval(qml.Z(1))

        We can print this QNode to show that the qfunc transform is taking place:

        >>> x = np.array(0.5, requires_grad=True)
        >>> y = np.array([0.1, 0.2], requires_grad=True)
        >>> print(qml.draw(circuit)(x, y))
        0: ──RX(0.10)──H────────╭Z─┤
        1: ──RX(0.05)──RY(0.14)─╰●─┤  <Z>

        Evaluating the QNode, as well as the derivative, with respect to the gate
        parameter *and* the transform weights:

        >>> circuit(x, y)
        tensor(0.98877939, requires_grad=True)
        >>> qml.grad(circuit)(x, y)
        (tensor(-0.02485651, requires_grad=True), array([-0.02474011, -0.09954244]))

        **Implementation details**

        Internally, the qfunc transform works as follows:

        .. code-block:: python

            def transform(old_qfunc, params):
                def new_qfunc(*args, **kwargs):
                    # 1. extract the QuantumTape from the old qfunc, being
                    # careful *not* to have it queued.
                    tape = make_qscript(old_qfunc)(*args, **kwargs)

                    # 2. transform the tape
                    new_tape = tape_transform(tape, params)

                    # 3. queue the *new* tape to the active queuing context
                    new_tape.queue()
                return new_qfunc

        *Note: this is pseudocode; the actual implementation is significantly more complicated!*

        Steps (1) and (3) are identical for all qfunc transforms; it is only step (2),
        ``tape_transform`` and the corresponding tape transform parameters, that define the qfunc
        transformation.

        That is, given a tape transform that **defines the qfunc transformation**, the
        decorator **elevates** the tape transform to one that works on quantum functions
        rather than tapes. This decorator therefore automates the process of adding in
        the queueing logic required under steps (1) and (3), so that it does not need to be
        repeated and tested for every new qfunc transform.
    """
    if os.environ.get('SPHINX_BUILD') == '1':
        warnings.warn("qfunc transformations have been disabled, as a Sphinx build has been detected via SPHINX_BUILD='1'. If this is not the case, please set the environment variable SPHINX_BUILD='0'.", UserWarning)
        return tape_transform
    if not callable(tape_transform):
        raise ValueError('The qfunc_transform decorator can only be applied to single tape transform functions.')
    warnings.warn('Use of `qfunc_transform` to create a custom transform is deprecated. Instead switch to using the new qml.transform function. Follow the instructions here for further details: https://docs.pennylane.ai/en/stable/code/qml_transforms.html#custom-transforms.', qml.PennyLaneDeprecationWarning)
    if not isinstance(tape_transform, single_tape_transform):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', qml.PennyLaneDeprecationWarning)
            tape_transform = single_tape_transform(tape_transform)
    sig = inspect.signature(tape_transform)
    params = sig.parameters
    if len(params) > 1:

        @functools.wraps(tape_transform)
        def make_qfunc_transform(*targs, **tkwargs):

            def wrapper(fn):
                return _create_qfunc_internal_wrapper(fn, tape_transform, targs, tkwargs)
            wrapper.tape_fn = functools.partial(tape_transform, *targs, **tkwargs)
            return wrapper
    elif len(params) == 1:

        @functools.wraps(tape_transform)
        def make_qfunc_transform(fn):
            return _create_qfunc_internal_wrapper(fn, tape_transform, [], {})
    make_qfunc_transform.tape_fn = tape_transform
    return make_qfunc_transform