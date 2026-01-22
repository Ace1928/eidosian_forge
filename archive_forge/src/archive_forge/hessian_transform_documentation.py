import warnings
from string import ascii_letters as ABC
import pennylane as qml
from pennylane.transforms.tape_expand import expand_invalid_trainable
Decorator for defining quantum Hessian transforms.

    Quantum Hessian transforms are a specific case of :class:`~.batch_transform`s,
    similar to the :class:`~.gradient_transform`. Hessian transforms compute the
    second derivative of a quantum function.
    All quantum Hessian transforms accept a tape, and output a batch of tapes to
    be independently executed on a quantum device, alongside a post-processing
    function to return the result.

    Args:
        expand_fn (function): An expansion function (if required) to be applied to the
            input tape before the Hessian computation takes place. If not provided,
            the default expansion function simply expands all operations that
            have ``Operation.grad_method=None`` until all resulting operations
            have a defined gradient method.
        differentiable (bool): Specifies whether the Hessian transform is differentiable
            or not. A transform may be non-differentiable if it does not use an autodiff
            framework for its tensor manipulations. In such a case, setting
            ``differentiable=False`` instructs the decorator to mark the output as
            'constant', reducing potential overhead.
        hybrid (bool): Specifies whether classical processing inside a QNode
            should be taken into account when transforming a QNode.

            - If ``True``, and classical processing is detected, the Jacobian of the
              classical processing will be computed and included. When evaluated, the
              returned Hessian will be with respect to the QNode arguments.

            - If ``False``, any internal QNode classical processing will be **ignored**.
              When evaluated, the returned Hessian will be with respect to the **gate**
              arguments, and not the QNode arguments.

    Supported Hessian transforms must be of the following form:

    .. code-block:: python

        @hessian_transform
        def my_custom_hessian(tape, **kwargs):
            ...
            return hessian_tapes, processing_fn

    where:

    - ``tape`` (*QuantumTape*): the input quantum tape to compute the Hessian of

    - ``hessian_tapes`` (*list[QuantumTape]*): is a list of output tapes to be
      evaluated. If this list is empty, no quantum evaluations will be made.

    - ``processing_fn`` is a processing function to be applied to the output of the
      evaluated ``hessian_tapes``. It should accept a list of numeric results with
      length ``len(hessian_tapes)``, and return the Hessian matrix.

    Once defined, the quantum Hessian transform can be used as follows:

    >>> hessian_tapes, processing_fn = my_custom_hessian(tape, *hessian_kwargs)
    >>> res = execute(tapes, dev, interface="autograd", gradient_fn=qml.gradients.param_shift)
    >>> jacobian = processing_fn(res)

    Alternatively, Hessian transforms can be applied directly to QNodes, in which case
    the execution is implicit:

    >>> fn = my_custom_hessian(qnode, *hessian_kwargs)
    >>> fn(weights)  # transformed function takes the same arguments as the QNode
    1.2629730888100839

    .. note::

        The input tape might have parameters of various types, including NumPy arrays,
        JAX Arrays, and TensorFlow and PyTorch tensors.

        If the Hessian transform is written in a autodiff-compatible manner, either by
        using a framework such as Autograd or TensorFlow, or by using ``qml.math`` for
        tensor manipulation, then higher-order derivatives will also be supported.

        Alternatively, you may use the ``tape.unwrap()`` context manager to temporarily
        convert all tape parameters to NumPy arrays and floats:

        >>> with tape.unwrap():
        ...     params = tape.get_parameters()  # list of floats
    