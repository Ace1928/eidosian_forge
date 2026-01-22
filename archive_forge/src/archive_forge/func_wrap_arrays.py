from collections.abc import Sequence
import functools
from autograd import numpy as _np
from .tensor import tensor
def wrap_arrays(old, new):
    """Loop through an object's symbol table,
    wrapping each function with :func:`~pennylane.numpy.tensor_wrapper`.

    This is useful if you would like to wrap **every** function
    provided by an imported module.

    Args:
        old (dict): The symbol table to be wrapped. Note that
            callable classes are ignored; only functions are wrapped.
        new (dict): The symbol table that contains the wrapped values.

    .. seealso:: :func:`~pennylane.numpy.tensor_wrapper`

    **Example**

    This function is used to wrap the imported ``autograd.numpy``
    module, to enable all functions to support ``requires_grad``
    arguments, and to output :class:`~pennylane.numpy.tensor` objects:

    >>> from autograd import numpy as _np
    >>> wrap_arrays(_np.__dict__, globals())
    """
    for name, obj in old.items():
        if callable(obj) and (not isinstance(obj, type)):
            new[name] = tensor_wrapper(obj)