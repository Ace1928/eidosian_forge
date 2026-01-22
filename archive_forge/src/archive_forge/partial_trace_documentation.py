from typing import Optional, Tuple
import numpy as np
import scipy.sparse as sp
from cvxpy.atoms.atom import Atom

    Assumes :math:`\texttt{expr} = X_1 \otimes \cdots \otimes X_n` is a 2D Kronecker
    product composed of :math:`n = \texttt{len(dims)}` implicit subsystems.
    Letting :math:`k = \texttt{axis}`, the returned expression represents
    the *partial trace* of :math:`\texttt{expr}` along its :math:`k^{\text{th}}` implicit subsystem:

    .. math::

        \text{tr}(X_k) (X_1 \otimes \cdots \otimes X_{k-1} \otimes X_{k+1} \otimes \cdots \otimes X_n).

    Parameters
    ----------
    expr : :class:`~cvxpy.expressions.expression.Expression`
        The 2D expression to take the partial trace of.
    dims : tuple of ints.
        A tuple of integers encoding the dimensions of each subsystem.
    axis : int
        The index of the subsystem to be traced out
        from the tensor product that defines expr.
    