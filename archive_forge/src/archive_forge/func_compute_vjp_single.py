import numpy as np
import autograd
import pennylane as qml
def compute_vjp_single(dy, jac, num=None):
    """Convenience function to compute the vector-Jacobian product for a given
    vector of gradient outputs and a Jacobian for a single measurement tape.

    Args:
        dy (tensor_like): vector of gradient outputs
        jac (tensor_like, tuple): Jacobian matrix
        num (int): The length of the flattened ``dy`` argument. This is an
            optional argument, but can be useful to provide if ``dy`` potentially
            has no shape (for example, due to tracing or just-in-time compilation).

    Returns:
        tensor_like: the vector-Jacobian product

    **Examples**

    1. For a single parameter and a single measurement without shape (e.g. expval, var):

    .. code-block:: pycon

        >>> jac = np.array(0.1)
        >>> dy = np.array(2)
        >>> compute_vjp_single(dy, jac)
        np.array([0.2])

    2. For a single parameter and a single measurement with shape (e.g. probs):

    .. code-block:: pycon

        >>> jac = np.array([0.1, 0.2])
        >>> dy = np.array([1.0, 1.0])
        >>> compute_vjp_single(dy, jac)
        np.array([0.3])


    3. For multiple parameters (in this case 2 parameters) and a single measurement without shape (e.g. expval, var):

    .. code-block:: pycon

        >>> jac = tuple([np.array(0.1), np.array(0.2)])
        >>> dy = np.array(2)
        >>> compute_vjp_single(dy, jac)
        np.array([0.2, 0.4])

    4. For multiple parameters (in this case 2 parameters) and a single measurement with shape (e.g. probs):

    .. code-block:: pycon

        >>> jac = tuple([np.array([0.1, 0.2]), np.array([0.3, 0.4])])
        >>> dy = np.array([1.0, 2.0])
        >>> compute_vjp_single(dy, jac)
        np.array([0.5, 1.1])

    """
    if jac is None:
        return None
    dy_row = qml.math.reshape(dy, [-1])
    if num is None:
        num = qml.math.shape(dy_row)[0]
    if not isinstance(dy_row, np.ndarray):
        jac = _convert(jac, dy_row)
    if not isinstance(jac, (tuple, autograd.builtins.SequenceBox)):
        if jac.shape == (0,):
            res = qml.math.zeros((1, 0))
            return res
        if num == 1:
            jac = qml.math.squeeze(jac)
        jac = qml.math.reshape(jac, (-1, 1))
        try:
            res = dy_row @ jac
        except Exception:
            res = qml.math.tensordot(jac, dy_row, [[0], [0]])
    else:
        if len(jac) == 0:
            res = qml.math.zeros((1, 0))
            return res
        if num == 1:
            jac = qml.math.reshape(qml.math.stack(jac), (1, -1))
            try:
                res = dy_row @ jac
            except Exception:
                res = qml.math.tensordot(jac, dy_row, [[0], [0]])
        else:
            jac = qml.math.stack(jac)
            try:
                res = jac @ dy_row
            except Exception:
                res = qml.math.tensordot(jac, dy_row, [[1], [0]])
    return res