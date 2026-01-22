from typing import Union
from copy import copy
import pennylane as qml
import pennylane.math as qnp
from pennylane.operation import Operator, convert_to_opmath
from pennylane.ops.op_math.pow import Pow
from pennylane.ops.op_math.sum import Sum
from pennylane.queuing import QueuingManager
from .symbolicop import ScalarSymbolicOp
def s_prod(scalar, operator, lazy=True, id=None):
    """Construct an operator which is the scalar product of the
    given scalar and operator provided.

    Args:
        scalar (float or complex): the scale factor being multiplied to the operator.
        operator (~.operation.Operator): the operator which will get scaled.

    Keyword Args:
        lazy=True (bool): If ``lazy=False`` and the operator is already a scalar product operator, the scalar provided will simply be combined with the existing scaling factor.
        id (str or None): id for the scalar product operator. Default is None.
    Returns:
        ~ops.op_math.SProd: The operator representing the scalar product.

    .. note::

        This operator supports a batched base, a batched coefficient and a combination of both:

        >>> op = qml.s_prod(scalar=4, operator=qml.RX([1, 2, 3], wires=0))
        >>> qml.matrix(op).shape
        (3, 2, 2)
        >>> op = qml.s_prod(scalar=[1, 2, 3], operator=qml.RX(1, wires=0))
        >>> qml.matrix(op).shape
        (3, 2, 2)
        >>> op = qml.s_prod(scalar=[4, 5, 6], operator=qml.RX([1, 2, 3], wires=0))
        >>> qml.matrix(op).shape
        (3, 2, 2)

        But it doesn't support batching of operators:

        >>> op = qml.s_prod(scalar=4, operator=[qml.RX(1, wires=0), qml.RX(2, wires=0)])
        AttributeError: 'list' object has no attribute 'batch_size'

    .. seealso:: :class:`~.ops.op_math.SProd` and :class:`~.ops.op_math.SymbolicOp`

    **Example**

    >>> sprod_op = s_prod(2.0, qml.X(0))
    >>> sprod_op
    2.0 * X(0)
    >>> sprod_op.matrix()
    array([[ 0., 2.],
           [ 2., 0.]])
    """
    operator = convert_to_opmath(operator)
    if lazy or not isinstance(operator, SProd):
        return SProd(scalar, operator, id=id)
    sprod_op = SProd(scalar=scalar * operator.scalar, base=operator.base, id=id)
    QueuingManager.remove(operator)
    return sprod_op