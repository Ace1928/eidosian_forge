from io import StringIO
from sympy.core import S, symbols, Eq, pi, Catalan, EulerGamma, Function
from sympy.core.relational import Equality
from sympy.functions.elementary.piecewise import Piecewise
from sympy.matrices import Matrix, MatrixSymbol
from sympy.utilities.codegen import OctaveCodeGen, codegen, make_routine
from sympy.testing.pytest import raises
from sympy.testing.pytest import XFAIL
import sympy
def test_m_filename_match_first_fcn():
    name_expr = [('foo', [2 * x, 3 * y]), ('bar', [y ** 2, 4 * y])]
    raises(ValueError, lambda: codegen(name_expr, 'Octave', prefix='bar', header=False, empty=False))