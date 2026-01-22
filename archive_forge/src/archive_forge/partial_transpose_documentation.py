from typing import Optional, Tuple
import numpy as np
import scipy.sparse as sp
from cvxpy.atoms.atom import Atom

    Assumes :math:`\texttt{expr} = X_1 \otimes ... \otimes X_n` is a 2D Kronecker
    product composed of :math:`n = \texttt{len(dims)}` implicit subsystems.
    Letting :math:`k = \texttt{axis}`, the returned expression is a
    *partial transpose* of :math:`\texttt{expr}`, with the transpose applied to its
    :math:`k^{\text{th}}` implicit subsystem:

    .. math::
        X_1 \otimes ... \otimes X_k^T \otimes ... \otimes X_n.

    Parameters
    ----------
    expr : :class:`~cvxpy.expressions.expression.Expression`
        The 2D expression to take the partial transpose of.
    dims : tuple of ints.
        A tuple of integers encoding the dimensions of each subsystem.
    axis : int
        The index of the subsystem to be transposed
        from the tensor product that defines expr.
    