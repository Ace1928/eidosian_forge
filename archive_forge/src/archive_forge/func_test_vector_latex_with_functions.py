from sympy.core.function import Function
from sympy.core.symbol import symbols
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (asin, cos, sin)
from sympy.physics.vector import ReferenceFrame, dynamicsymbols, Dyadic
from sympy.physics.vector.printing import (VectorLatexPrinter, vpprint,
def test_vector_latex_with_functions():
    N = ReferenceFrame('N')
    omega, alpha = dynamicsymbols('omega, alpha')
    v = omega.diff() * N.x
    assert vlatex(v) == '\\dot{\\omega}\\mathbf{\\hat{n}_x}'
    v = omega.diff() ** alpha * N.x
    assert vlatex(v) == '\\dot{\\omega}^{\\alpha}\\mathbf{\\hat{n}_x}'