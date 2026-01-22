from sympy.core.function import Function
from sympy.core.symbol import symbols
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (asin, cos, sin)
from sympy.physics.vector import ReferenceFrame, dynamicsymbols, Dyadic
from sympy.physics.vector.printing import (VectorLatexPrinter, vpprint,
def test_vector_str_arguments():
    assert vsprint(N.x * 3.0, full_prec=False) == '3.0*N.x'
    assert vsprint(N.x * 3.0, full_prec=True) == '3.00000000000000*N.x'