from sympy.core.function import Function
from sympy.core.symbol import symbols
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (asin, cos, sin)
from sympy.physics.vector import ReferenceFrame, dynamicsymbols, Dyadic
from sympy.physics.vector.printing import (VectorLatexPrinter, vpprint,
def test_vector_derivative_printing():
    v = omega.diff() * N.x
    assert unicode_vpretty(v) == 'ω̇ n_x'
    assert ascii_vpretty(v) == "omega'(t) n_x"
    v = omega.diff().diff() * N.x
    assert vlatex(v) == '\\ddot{\\omega}\\mathbf{\\hat{n}_x}'
    assert unicode_vpretty(v) == 'ω̈ n_x'
    assert ascii_vpretty(v) == "omega''(t) n_x"
    v = omega.diff().diff().diff() * N.x
    assert vlatex(v) == '\\dddot{\\omega}\\mathbf{\\hat{n}_x}'
    assert unicode_vpretty(v) == 'ω⃛ n_x'
    assert ascii_vpretty(v) == "omega'''(t) n_x"
    v = omega.diff().diff().diff().diff() * N.x
    assert vlatex(v) == '\\ddddot{\\omega}\\mathbf{\\hat{n}_x}'
    assert unicode_vpretty(v) == 'ω⃜ n_x'
    assert ascii_vpretty(v) == "omega''''(t) n_x"
    v = omega.diff().diff().diff().diff().diff() * N.x
    assert vlatex(v) == '\\frac{d^{5}}{d t^{5}} \\omega\\mathbf{\\hat{n}_x}'
    assert unicode_vpretty(v) == '  5\n d\n───(ω) n_x\n  5\ndt'
    assert ascii_vpretty(v) == '  5\n d\n---(omega) n_x\n  5\ndt'