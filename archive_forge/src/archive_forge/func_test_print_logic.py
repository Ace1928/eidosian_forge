from sympy.calculus.accumulationbounds import AccumBounds
from sympy.concrete.summations import Sum
from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.function import Derivative, Lambda, diff, Function
from sympy.core.numbers import (zoo, Float, Integer, I, oo, pi, E,
from sympy.core.relational import Lt, Ge, Ne, Eq
from sympy.core.singleton import S
from sympy.core.symbol import symbols, Symbol
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import (factorial2,
from sympy.functions.combinatorial.numbers import (lucas, bell,
from sympy.functions.elementary.complexes import re, im, conjugate, Abs
from sympy.functions.elementary.exponential import exp, LambertW, log
from sympy.functions.elementary.hyperbolic import (tanh, acoth, atanh,
from sympy.functions.elementary.integers import ceiling, floor
from sympy.functions.elementary.miscellaneous import Max, Min
from sympy.functions.elementary.trigonometric import (csc, sec, tan,
from sympy.functions.special.delta_functions import Heaviside
from sympy.functions.special.elliptic_integrals import (elliptic_pi,
from sympy.functions.special.error_functions import (fresnelc,
from sympy.functions.special.gamma_functions import (gamma, uppergamma,
from sympy.functions.special.mathieu_functions import (mathieusprime,
from sympy.functions.special.polynomials import (jacobi, chebyshevu,
from sympy.functions.special.singularity_functions import SingularityFunction
from sympy.functions.special.zeta_functions import (polylog, stieltjes,
from sympy.integrals.integrals import Integral
from sympy.logic.boolalg import (Xor, Or, false, true, And, Equivalent,
from sympy.matrices.dense import Matrix
from sympy.matrices.expressions.determinant import Determinant
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.ntheory.factor_ import (totient, reduced_totient, primenu,
from sympy.physics.quantum import (ComplexSpace, FockSpace, hbar,
from sympy.printing.mathml import (MathMLPresentationPrinter,
from sympy.series.limits import Limit
from sympy.sets.contains import Contains
from sympy.sets.fancysets import Range
from sympy.sets.sets import (Interval, Union, SymmetricDifference,
from sympy.stats.rv import RandomSymbol
from sympy.tensor.indexed import IndexedBase
from sympy.vector import (Divergence, CoordSys3D, Cross, Curl, Dot,
from sympy.testing.pytest import raises
def test_print_logic():
    assert mpp.doprint(And(x, y)) == '<mrow><mi>x</mi><mo>&#x2227;</mo><mi>y</mi></mrow>'
    assert mpp.doprint(Or(x, y)) == '<mrow><mi>x</mi><mo>&#x2228;</mo><mi>y</mi></mrow>'
    assert mpp.doprint(Xor(x, y)) == '<mrow><mi>x</mi><mo>&#x22BB;</mo><mi>y</mi></mrow>'
    assert mpp.doprint(Implies(x, y)) == '<mrow><mi>x</mi><mo>&#x21D2;</mo><mi>y</mi></mrow>'
    assert mpp.doprint(Equivalent(x, y)) == '<mrow><mi>x</mi><mo>&#x21D4;</mo><mi>y</mi></mrow>'
    assert mpp.doprint(And(Eq(x, y), x > 4)) == '<mrow><mrow><mi>x</mi><mo>=</mo><mi>y</mi></mrow><mo>&#x2227;</mo><mrow><mi>x</mi><mo>></mo><mn>4</mn></mrow></mrow>'
    assert mpp.doprint(And(Eq(x, 3), y < 3, x > y + 1)) == '<mrow><mrow><mi>x</mi><mo>=</mo><mn>3</mn></mrow><mo>&#x2227;</mo><mrow><mi>x</mi><mo>></mo><mrow><mi>y</mi><mo>+</mo><mn>1</mn></mrow></mrow><mo>&#x2227;</mo><mrow><mi>y</mi><mo><</mo><mn>3</mn></mrow></mrow>'
    assert mpp.doprint(Or(Eq(x, y), x > 4)) == '<mrow><mrow><mi>x</mi><mo>=</mo><mi>y</mi></mrow><mo>&#x2228;</mo><mrow><mi>x</mi><mo>></mo><mn>4</mn></mrow></mrow>'
    assert mpp.doprint(And(Eq(x, 3), Or(y < 3, x > y + 1))) == '<mrow><mrow><mi>x</mi><mo>=</mo><mn>3</mn></mrow><mo>&#x2227;</mo><mfenced><mrow><mrow><mi>x</mi><mo>></mo><mrow><mi>y</mi><mo>+</mo><mn>1</mn></mrow></mrow><mo>&#x2228;</mo><mrow><mi>y</mi><mo><</mo><mn>3</mn></mrow></mrow></mfenced></mrow>'
    assert mpp.doprint(Not(x)) == '<mrow><mo>&#xAC;</mo><mi>x</mi></mrow>'
    assert mpp.doprint(Not(And(x, y))) == '<mrow><mo>&#xAC;</mo><mfenced><mrow><mi>x</mi><mo>&#x2227;</mo><mi>y</mi></mrow></mfenced></mrow>'