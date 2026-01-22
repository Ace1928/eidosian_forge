from sympy.core.function import Function
from sympy.core.symbol import symbols
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (asin, cos, sin)
from sympy.physics.vector import ReferenceFrame, dynamicsymbols, Dyadic
from sympy.physics.vector.printing import (VectorLatexPrinter, vpprint,
def test_vector_latex():
    a, b, c, d, omega = symbols('a, b, c, d, omega')
    v = (a ** 2 + b / c) * A.x + sqrt(d) * A.y + cos(omega) * A.z
    assert vlatex(v) == '(a^{2} + \\frac{b}{c})\\mathbf{\\hat{a}_x} + \\sqrt{d}\\mathbf{\\hat{a}_y} + \\cos{\\left(\\omega \\right)}\\mathbf{\\hat{a}_z}'
    theta, omega, alpha, q = dynamicsymbols('theta, omega, alpha, q')
    v = theta * A.x + omega * omega * A.y + q * alpha * A.z
    assert vlatex(v) == '\\theta\\mathbf{\\hat{a}_x} + \\omega^{2}\\mathbf{\\hat{a}_y} + \\alpha q\\mathbf{\\hat{a}_z}'
    phi1, phi2, phi3 = dynamicsymbols('phi1, phi2, phi3')
    theta1, theta2, theta3 = symbols('theta1, theta2, theta3')
    v = sin(theta1) * A.x + cos(phi1) * cos(phi2) * A.y + cos(theta1 + phi3) * A.z
    assert vlatex(v) == '\\sin{\\left(\\theta_{1} \\right)}\\mathbf{\\hat{a}_x} + \\cos{\\left(\\phi_{1} \\right)} \\cos{\\left(\\phi_{2} \\right)}\\mathbf{\\hat{a}_y} + \\cos{\\left(\\theta_{1} + \\phi_{3} \\right)}\\mathbf{\\hat{a}_z}'
    N = ReferenceFrame('N')
    a, b, c, d, omega = symbols('a, b, c, d, omega')
    v = (a ** 2 + b / c) * N.x + sqrt(d) * N.y + cos(omega) * N.z
    expected = '(a^{2} + \\frac{b}{c})\\mathbf{\\hat{n}_x} + \\sqrt{d}\\mathbf{\\hat{n}_y} + \\cos{\\left(\\omega \\right)}\\mathbf{\\hat{n}_z}'
    assert vlatex(v) == expected
    N = ReferenceFrame('N', latexs=('\\hat{i}', '\\hat{j}', '\\hat{k}'))
    v = (a ** 2 + b / c) * N.x + sqrt(d) * N.y + cos(omega) * N.z
    expected = '(a^{2} + \\frac{b}{c})\\hat{i} + \\sqrt{d}\\hat{j} + \\cos{\\left(\\omega \\right)}\\hat{k}'
    assert vlatex(v) == expected
    expected = '\\alpha\\mathbf{\\hat{n}_x} + \\operatorname{asin}{\\left(\\omega \\right)}\\mathbf{\\hat{n}_y} -  \\beta \\dot{\\alpha}\\mathbf{\\hat{n}_z}'
    assert vlatex(ww) == expected
    expected = '- \\mathbf{\\hat{n}_x}\\otimes \\mathbf{\\hat{n}_y} - \\mathbf{\\hat{n}_x}\\otimes \\mathbf{\\hat{n}_z}'
    assert vlatex(xx) == expected
    expected = '\\mathbf{\\hat{n}_x}\\otimes \\mathbf{\\hat{n}_y} + \\mathbf{\\hat{n}_x}\\otimes \\mathbf{\\hat{n}_z}'
    assert vlatex(xx2) == expected