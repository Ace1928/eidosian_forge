from array import array
import ctypes
import functools
from ..base import _LIB, check_call, string_types
from ..base import mx_uint, NDArrayHandle, c_array, c_array_buf, c_handle_array
from ..ndarray import NDArray, zeros_like, _GRAD_REQ_MAP
def grad_and_loss(func, argnum=None):
    """Return function that computes both gradient of arguments and loss value.

    Parameters
    ----------
    func: a python function
        The forward (loss) function.
    argnum: an int or a list of int
        The index of argument to calculate gradient for.

    Returns
    -------
    grad_and_loss_func: a python function
        A function that would compute both the gradient of arguments and loss value.
    """

    @functools.wraps(func)
    def wrapped(*args):
        """Wrapped function."""
        variables = args
        if argnum is not None:
            argnum_ = argnum if isinstance(argnum, list) else [argnum]
            variables = [args[i] for i in argnum_]
        for x in variables:
            assert isinstance(x, NDArray), 'type of autograd input should NDArray.'
        grads = [zeros_like(x) for x in variables]
        mark_variables(variables, grads)
        with train_section():
            outputs = func(*args)
        compute_gradient([outputs] if isinstance(outputs, NDArray) else outputs)
        return (grads, outputs)
    return wrapped