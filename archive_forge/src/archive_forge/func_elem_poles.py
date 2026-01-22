from typing import Type
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.evalf import EvalfMixin
from sympy.core.expr import Expr
from sympy.core.function import expand
from sympy.core.logic import fuzzy_and
from sympy.core.mul import Mul
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol
from sympy.core.sympify import sympify, _sympify
from sympy.matrices import ImmutableMatrix, eye
from sympy.matrices.expressions import MatMul, MatAdd
from sympy.polys import Poly, rootof
from sympy.polys.polyroots import roots
from sympy.polys.polytools import (cancel, degree)
from sympy.series import limit
from mpmath.libmp.libmpf import prec_to_dps
def elem_poles(self):
    """
        Returns the poles of each element of the ``TransferFunctionMatrix``.

        .. note::
            Actual poles of a MIMO system are NOT the poles of individual elements.

        Examples
        ========

        >>> from sympy.abc import s
        >>> from sympy.physics.control.lti import TransferFunction, TransferFunctionMatrix
        >>> tf_1 = TransferFunction(3, (s + 1), s)
        >>> tf_2 = TransferFunction(s + 6, (s + 1)*(s + 2), s)
        >>> tf_3 = TransferFunction(s + 3, s**2 + 3*s + 2, s)
        >>> tf_4 = TransferFunction(s + 2, s**2 + 5*s - 10, s)
        >>> tfm_1 = TransferFunctionMatrix([[tf_1, tf_2], [tf_3, tf_4]])
        >>> tfm_1
        TransferFunctionMatrix(((TransferFunction(3, s + 1, s), TransferFunction(s + 6, (s + 1)*(s + 2), s)), (TransferFunction(s + 3, s**2 + 3*s + 2, s), TransferFunction(s + 2, s**2 + 5*s - 10, s))))
        >>> tfm_1.elem_poles()
        [[[-1], [-2, -1]], [[-2, -1], [-5/2 + sqrt(65)/2, -sqrt(65)/2 - 5/2]]]

        See Also
        ========

        elem_zeros

        """
    return [[element.poles() for element in row] for row in self.doit().args[0]]