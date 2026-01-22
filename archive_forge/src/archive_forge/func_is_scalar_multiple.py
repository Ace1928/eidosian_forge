import warnings
from sympy.core import S, sympify, Expr
from sympy.core.add import Add
from sympy.core.containers import Tuple
from sympy.core.numbers import Float
from sympy.core.parameters import global_parameters
from sympy.simplify import nsimplify, simplify
from sympy.geometry.exceptions import GeometryError
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.complexes import im
from sympy.functions.elementary.trigonometric import cos, sin
from sympy.matrices import Matrix
from sympy.matrices.expressions import Transpose
from sympy.utilities.iterables import uniq, is_sequence
from sympy.utilities.misc import filldedent, func_name, Undecidable
from .entity import GeometryEntity
from mpmath.libmp.libmpf import prec_to_dps
def is_scalar_multiple(self, p):
    """Returns whether each coordinate of `self` is a scalar
        multiple of the corresponding coordinate in point p.
        """
    s, o = Point._normalize_dimension(self, Point(p))
    if s.ambient_dimension == 2:
        (x1, y1), (x2, y2) = (s.args, o.args)
        rv = (x1 * y2 - x2 * y1).equals(0)
        if rv is None:
            raise Undecidable(filldedent('Cannot determine if %s is a scalar multiple of\n                    %s' % (s, o)))
    m = Matrix([s.args, o.args])
    return m.rank() < 2