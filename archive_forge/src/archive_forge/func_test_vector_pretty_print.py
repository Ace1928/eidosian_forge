from sympy.core.function import Function
from sympy.core.symbol import symbols
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (asin, cos, sin)
from sympy.physics.vector import ReferenceFrame, dynamicsymbols, Dyadic
from sympy.physics.vector.printing import (VectorLatexPrinter, vpprint,
def test_vector_pretty_print():
    expected = ' 2\na  n_x + b n_y + c*sin(alpha) n_z'
    uexpected = ' 2\na  n_x + b n_y + c⋅sin(α) n_z'
    assert ascii_vpretty(v) == expected
    assert unicode_vpretty(v) == uexpected
    expected = 'alpha n_x + sin(omega) n_y + alpha*beta n_z'
    uexpected = 'α n_x + sin(ω) n_y + α⋅β n_z'
    assert ascii_vpretty(w) == expected
    assert unicode_vpretty(w) == uexpected
    expected = '                     2\na       b + c       c\n- n_x + ----- n_y + -- n_z\nb         a         b'
    uexpected = '                     2\na       b + c       c\n─ n_x + ───── n_y + ── n_z\nb         a         b'
    assert ascii_vpretty(o) == expected
    assert unicode_vpretty(o) == uexpected