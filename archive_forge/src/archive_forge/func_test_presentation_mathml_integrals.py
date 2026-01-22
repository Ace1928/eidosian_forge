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
def test_presentation_mathml_integrals():
    assert mpp.doprint(Integral(x, (x, 0, 1))) == '<mrow><msubsup><mo>&#x222B;</mo><mn>0</mn><mn>1</mn></msubsup><mi>x</mi><mo>&dd;</mo><mi>x</mi></mrow>'
    assert mpp.doprint(Integral(log(x), x)) == '<mrow><mo>&#x222B;</mo><mrow><mi>log</mi><mfenced><mi>x</mi></mfenced></mrow><mo>&dd;</mo><mi>x</mi></mrow>'
    assert mpp.doprint(Integral(x * y, x, y)) == '<mrow><mo>&#x222C;</mo><mrow><mi>x</mi><mo>&InvisibleTimes;</mo><mi>y</mi></mrow><mo>&dd;</mo><mi>y</mi><mo>&dd;</mo><mi>x</mi></mrow>'
    z, w = symbols('z w')
    assert mpp.doprint(Integral(x * y * z, x, y, z)) == '<mrow><mo>&#x222D;</mo><mrow><mi>x</mi><mo>&InvisibleTimes;</mo><mi>y</mi><mo>&InvisibleTimes;</mo><mi>z</mi></mrow><mo>&dd;</mo><mi>z</mi><mo>&dd;</mo><mi>y</mi><mo>&dd;</mo><mi>x</mi></mrow>'
    assert mpp.doprint(Integral(x * y * z * w, x, y, z, w)) == '<mrow><mo>&#x222B;</mo><mo>&#x222B;</mo><mo>&#x222B;</mo><mo>&#x222B;</mo><mrow><mi>w</mi><mo>&InvisibleTimes;</mo><mi>x</mi><mo>&InvisibleTimes;</mo><mi>y</mi><mo>&InvisibleTimes;</mo><mi>z</mi></mrow><mo>&dd;</mo><mi>w</mi><mo>&dd;</mo><mi>z</mi><mo>&dd;</mo><mi>y</mi><mo>&dd;</mo><mi>x</mi></mrow>'
    assert mpp.doprint(Integral(x, x, y, (z, 0, 1))) == '<mrow><msubsup><mo>&#x222B;</mo><mn>0</mn><mn>1</mn></msubsup><mo>&#x222B;</mo><mo>&#x222B;</mo><mi>x</mi><mo>&dd;</mo><mi>z</mi><mo>&dd;</mo><mi>y</mi><mo>&dd;</mo><mi>x</mi></mrow>'
    assert mpp.doprint(Integral(x, (x, 0))) == '<mrow><msup><mo>&#x222B;</mo><mn>0</mn></msup><mi>x</mi><mo>&dd;</mo><mi>x</mi></mrow>'