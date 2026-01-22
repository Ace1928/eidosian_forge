import functools
from collections.abc import Sequence
import autoray as ar
import numpy as onp
from autograd.numpy.numpy_boxes import ArrayBox
from autoray import numpy as np
from numpy import ndarray
from . import single_dispatch  # pylint:disable=unused-import
from .utils import cast, cast_like, get_interface, requires_grad
def multi_dispatch(argnum=None, tensor_list=None):
    """Decorater to dispatch arguments handled by the interface.

    This helps simplify definitions of new functions inside PennyLane. We can
    decorate the function, indicating the arguments that are tensors handled
    by the interface:


    >>> @qml.math.multi_dispatch(argnum=[0, 1])
    ... def some_function(tensor1, tensor2, option, like):
    ...     # the interface string is stored in `like`.
    ...     ...


    Args:
        argnum (list[int]): A list of integers indicating the indices
            to dispatch (i.e., the arguments that are tensors handled by an interface).
            If ``None``, dispatch over all arguments.
        tensor_lists (list[int]): a list of integers indicating which indices
            in ``argnum`` are expected to be lists of tensors. If an argument
            marked as tensor list is not a ``tuple`` or ``list``, it is treated
            as if it was not marked as tensor list. If ``None``, this option is ignored.

    Returns:
        func: A wrapped version of the function, which will automatically attempt
        to dispatch to the correct autodifferentiation framework for the requested
        arguments. Note that the ``like`` argument will be optional, but can be provided
        if an explicit override is needed.

    .. seealso:: :func:`pennylane.math.multi_dispatch._multi_dispatch`

    .. note::
        This decorator makes the interface argument "like" optional as it utilizes
        the utility function `_multi_dispatch` to automatically detect the appropriate
        interface based on the tensor types.

    **Examples**

    We can redefine external functions to be suitable for PennyLane. Here, we
    redefine Autoray's ``stack`` function.

    >>> stack = multi_dispatch(argnum=0, tensor_list=0)(autoray.numpy.stack)

    We can also use the ``multi_dispatch`` decorator to dispatch
    arguments of more more elaborate custom functions. Here is an example
    of a ``custom_function`` that
    computes :math:`c \\\\sum_i (v_i)^T v_i`, where :math:`v_i` are vectors in ``values`` and
    :math:`c` is a fixed ``coefficient``. Note how ``argnum=0`` only points to the first argument ``values``,
    how ``tensor_list=0`` indicates that said first argument is a list of vectors, and that ``coefficient`` is not
    dispatched.

    >>> @math.multi_dispatch(argnum=0, tensor_list=0)
    >>> def custom_function(values, like, coefficient=10):
    >>>     # values is a list of vectors
    >>>     # like can force the interface (optional)
    >>>     if like == "tensorflow":
    >>>         # add interface-specific handling if necessary
    >>>     return coefficient * np.sum([math.dot(v,v) for v in values])

    We can then run

    >>> values = [np.array([1, 2, 3]) for _ in range(5)]
    >>> custom_function(values)
    700

    """

    def decorator(fn):

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            argnums = argnum if argnum is not None else list(range(len(args)))
            tensor_lists = tensor_list if tensor_list is not None else []
            if not isinstance(argnums, Sequence):
                argnums = [argnums]
            if not isinstance(tensor_lists, Sequence):
                tensor_lists = [tensor_lists]
            dispatch_args = []
            for a in argnums:
                if a in tensor_lists and isinstance(args[a], (list, tuple)):
                    dispatch_args.extend(args[a])
                else:
                    dispatch_args.append(args[a])
            interface = kwargs.pop('like', None)
            interface = interface or get_interface(*dispatch_args)
            kwargs['like'] = interface
            return fn(*args, **kwargs)
        return wrapper
    return decorator