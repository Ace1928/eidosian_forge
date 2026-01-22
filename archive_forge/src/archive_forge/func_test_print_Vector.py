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
def test_print_Vector():
    ACS = CoordSys3D('A')
    assert mathml(Cross(ACS.i, ACS.j * ACS.x * 3 + ACS.k), printer='presentation') == '<mrow><msub><mover><mi mathvariant="bold">i</mi><mo>^</mo></mover><mi mathvariant="bold">A</mi></msub><mo>&#xD7;</mo><mfenced><mrow><mfenced><mrow><mn>3</mn><mo>&InvisibleTimes;</mo><msub><mi mathvariant="bold">x</mi><mi mathvariant="bold">A</mi></msub></mrow></mfenced><mo>&InvisibleTimes;</mo><msub><mover><mi mathvariant="bold">j</mi><mo>^</mo></mover><mi mathvariant="bold">A</mi></msub><mo>+</mo><msub><mover><mi mathvariant="bold">k</mi><mo>^</mo></mover><mi mathvariant="bold">A</mi></msub></mrow></mfenced></mrow>'
    assert mathml(Cross(ACS.i, ACS.j), printer='presentation') == '<mrow><msub><mover><mi mathvariant="bold">i</mi><mo>^</mo></mover><mi mathvariant="bold">A</mi></msub><mo>&#xD7;</mo><msub><mover><mi mathvariant="bold">j</mi><mo>^</mo></mover><mi mathvariant="bold">A</mi></msub></mrow>'
    assert mathml(x * Cross(ACS.i, ACS.j), printer='presentation') == '<mrow><mi>x</mi><mo>&InvisibleTimes;</mo><mfenced><mrow><msub><mover><mi mathvariant="bold">i</mi><mo>^</mo></mover><mi mathvariant="bold">A</mi></msub><mo>&#xD7;</mo><msub><mover><mi mathvariant="bold">j</mi><mo>^</mo></mover><mi mathvariant="bold">A</mi></msub></mrow></mfenced></mrow>'
    assert mathml(Cross(x * ACS.i, ACS.j), printer='presentation') == '<mrow><mo>-</mo><mrow><msub><mover><mi mathvariant="bold">j</mi><mo>^</mo></mover><mi mathvariant="bold">A</mi></msub><mo>&#xD7;</mo><mfenced><mrow><mfenced><mi>x</mi></mfenced><mo>&InvisibleTimes;</mo><msub><mover><mi mathvariant="bold">i</mi><mo>^</mo></mover><mi mathvariant="bold">A</mi></msub></mrow></mfenced></mrow></mrow>'
    assert mathml(Curl(3 * ACS.x * ACS.j), printer='presentation') == '<mrow><mo>&#x2207;</mo><mo>&#xD7;</mo><mfenced><mrow><mfenced><mrow><mn>3</mn><mo>&InvisibleTimes;</mo><msub><mi mathvariant="bold">x</mi><mi mathvariant="bold">A</mi></msub></mrow></mfenced><mo>&InvisibleTimes;</mo><msub><mover><mi mathvariant="bold">j</mi><mo>^</mo></mover><mi mathvariant="bold">A</mi></msub></mrow></mfenced></mrow>'
    assert mathml(Curl(3 * x * ACS.x * ACS.j), printer='presentation') == '<mrow><mo>&#x2207;</mo><mo>&#xD7;</mo><mfenced><mrow><mfenced><mrow><mn>3</mn><mo>&InvisibleTimes;</mo><msub><mi mathvariant="bold">x</mi><mi mathvariant="bold">A</mi></msub><mo>&InvisibleTimes;</mo><mi>x</mi></mrow></mfenced><mo>&InvisibleTimes;</mo><msub><mover><mi mathvariant="bold">j</mi><mo>^</mo></mover><mi mathvariant="bold">A</mi></msub></mrow></mfenced></mrow>'
    assert mathml(x * Curl(3 * ACS.x * ACS.j), printer='presentation') == '<mrow><mi>x</mi><mo>&InvisibleTimes;</mo><mfenced><mrow><mo>&#x2207;</mo><mo>&#xD7;</mo><mfenced><mrow><mfenced><mrow><mn>3</mn><mo>&InvisibleTimes;</mo><msub><mi mathvariant="bold">x</mi><mi mathvariant="bold">A</mi></msub></mrow></mfenced><mo>&InvisibleTimes;</mo><msub><mover><mi mathvariant="bold">j</mi><mo>^</mo></mover><mi mathvariant="bold">A</mi></msub></mrow></mfenced></mrow></mfenced></mrow>'
    assert mathml(Curl(3 * x * ACS.x * ACS.j + ACS.i), printer='presentation') == '<mrow><mo>&#x2207;</mo><mo>&#xD7;</mo><mfenced><mrow><msub><mover><mi mathvariant="bold">i</mi><mo>^</mo></mover><mi mathvariant="bold">A</mi></msub><mo>+</mo><mfenced><mrow><mn>3</mn><mo>&InvisibleTimes;</mo><msub><mi mathvariant="bold">x</mi><mi mathvariant="bold">A</mi></msub><mo>&InvisibleTimes;</mo><mi>x</mi></mrow></mfenced><mo>&InvisibleTimes;</mo><msub><mover><mi mathvariant="bold">j</mi><mo>^</mo></mover><mi mathvariant="bold">A</mi></msub></mrow></mfenced></mrow>'
    assert mathml(Divergence(3 * ACS.x * ACS.j), printer='presentation') == '<mrow><mo>&#x2207;</mo><mo>&#xB7;</mo><mfenced><mrow><mfenced><mrow><mn>3</mn><mo>&InvisibleTimes;</mo><msub><mi mathvariant="bold">x</mi><mi mathvariant="bold">A</mi></msub></mrow></mfenced><mo>&InvisibleTimes;</mo><msub><mover><mi mathvariant="bold">j</mi><mo>^</mo></mover><mi mathvariant="bold">A</mi></msub></mrow></mfenced></mrow>'
    assert mathml(x * Divergence(3 * ACS.x * ACS.j), printer='presentation') == '<mrow><mi>x</mi><mo>&InvisibleTimes;</mo><mfenced><mrow><mo>&#x2207;</mo><mo>&#xB7;</mo><mfenced><mrow><mfenced><mrow><mn>3</mn><mo>&InvisibleTimes;</mo><msub><mi mathvariant="bold">x</mi><mi mathvariant="bold">A</mi></msub></mrow></mfenced><mo>&InvisibleTimes;</mo><msub><mover><mi mathvariant="bold">j</mi><mo>^</mo></mover><mi mathvariant="bold">A</mi></msub></mrow></mfenced></mrow></mfenced></mrow>'
    assert mathml(Divergence(3 * x * ACS.x * ACS.j + ACS.i), printer='presentation') == '<mrow><mo>&#x2207;</mo><mo>&#xB7;</mo><mfenced><mrow><msub><mover><mi mathvariant="bold">i</mi><mo>^</mo></mover><mi mathvariant="bold">A</mi></msub><mo>+</mo><mfenced><mrow><mn>3</mn><mo>&InvisibleTimes;</mo><msub><mi mathvariant="bold">x</mi><mi mathvariant="bold">A</mi></msub><mo>&InvisibleTimes;</mo><mi>x</mi></mrow></mfenced><mo>&InvisibleTimes;</mo><msub><mover><mi mathvariant="bold">j</mi><mo>^</mo></mover><mi mathvariant="bold">A</mi></msub></mrow></mfenced></mrow>'
    assert mathml(Dot(ACS.i, ACS.j * ACS.x * 3 + ACS.k), printer='presentation') == '<mrow><msub><mover><mi mathvariant="bold">i</mi><mo>^</mo></mover><mi mathvariant="bold">A</mi></msub><mo>&#xB7;</mo><mfenced><mrow><mfenced><mrow><mn>3</mn><mo>&InvisibleTimes;</mo><msub><mi mathvariant="bold">x</mi><mi mathvariant="bold">A</mi></msub></mrow></mfenced><mo>&InvisibleTimes;</mo><msub><mover><mi mathvariant="bold">j</mi><mo>^</mo></mover><mi mathvariant="bold">A</mi></msub><mo>+</mo><msub><mover><mi mathvariant="bold">k</mi><mo>^</mo></mover><mi mathvariant="bold">A</mi></msub></mrow></mfenced></mrow>'
    assert mathml(Dot(ACS.i, ACS.j), printer='presentation') == '<mrow><msub><mover><mi mathvariant="bold">i</mi><mo>^</mo></mover><mi mathvariant="bold">A</mi></msub><mo>&#xB7;</mo><msub><mover><mi mathvariant="bold">j</mi><mo>^</mo></mover><mi mathvariant="bold">A</mi></msub></mrow>'
    assert mathml(Dot(x * ACS.i, ACS.j), printer='presentation') == '<mrow><msub><mover><mi mathvariant="bold">j</mi><mo>^</mo></mover><mi mathvariant="bold">A</mi></msub><mo>&#xB7;</mo><mfenced><mrow><mfenced><mi>x</mi></mfenced><mo>&InvisibleTimes;</mo><msub><mover><mi mathvariant="bold">i</mi><mo>^</mo></mover><mi mathvariant="bold">A</mi></msub></mrow></mfenced></mrow>'
    assert mathml(x * Dot(ACS.i, ACS.j), printer='presentation') == '<mrow><mi>x</mi><mo>&InvisibleTimes;</mo><mfenced><mrow><msub><mover><mi mathvariant="bold">i</mi><mo>^</mo></mover><mi mathvariant="bold">A</mi></msub><mo>&#xB7;</mo><msub><mover><mi mathvariant="bold">j</mi><mo>^</mo></mover><mi mathvariant="bold">A</mi></msub></mrow></mfenced></mrow>'
    assert mathml(Gradient(ACS.x), printer='presentation') == '<mrow><mo>&#x2207;</mo><msub><mi mathvariant="bold">x</mi><mi mathvariant="bold">A</mi></msub></mrow>'
    assert mathml(Gradient(ACS.x + 3 * ACS.y), printer='presentation') == '<mrow><mo>&#x2207;</mo><mfenced><mrow><msub><mi mathvariant="bold">x</mi><mi mathvariant="bold">A</mi></msub><mo>+</mo><mrow><mn>3</mn><mo>&InvisibleTimes;</mo><msub><mi mathvariant="bold">y</mi><mi mathvariant="bold">A</mi></msub></mrow></mrow></mfenced></mrow>'
    assert mathml(x * Gradient(ACS.x), printer='presentation') == '<mrow><mi>x</mi><mo>&InvisibleTimes;</mo><mfenced><mrow><mo>&#x2207;</mo><msub><mi mathvariant="bold">x</mi><mi mathvariant="bold">A</mi></msub></mrow></mfenced></mrow>'
    assert mathml(Gradient(x * ACS.x), printer='presentation') == '<mrow><mo>&#x2207;</mo><mfenced><mrow><msub><mi mathvariant="bold">x</mi><mi mathvariant="bold">A</mi></msub><mo>&InvisibleTimes;</mo><mi>x</mi></mrow></mfenced></mrow>'
    assert mathml(Cross(ACS.x, ACS.z) + Cross(ACS.z, ACS.x), printer='presentation') == '<mover><mi mathvariant="bold">0</mi><mo>^</mo></mover>'
    assert mathml(Cross(ACS.z, ACS.x), printer='presentation') == '<mrow><mo>-</mo><mrow><msub><mi mathvariant="bold">x</mi><mi mathvariant="bold">A</mi></msub><mo>&#xD7;</mo><msub><mi mathvariant="bold">z</mi><mi mathvariant="bold">A</mi></msub></mrow></mrow>'
    assert mathml(Laplacian(ACS.x), printer='presentation') == '<mrow><mo>&#x2206;</mo><msub><mi mathvariant="bold">x</mi><mi mathvariant="bold">A</mi></msub></mrow>'
    assert mathml(Laplacian(ACS.x + 3 * ACS.y), printer='presentation') == '<mrow><mo>&#x2206;</mo><mfenced><mrow><msub><mi mathvariant="bold">x</mi><mi mathvariant="bold">A</mi></msub><mo>+</mo><mrow><mn>3</mn><mo>&InvisibleTimes;</mo><msub><mi mathvariant="bold">y</mi><mi mathvariant="bold">A</mi></msub></mrow></mrow></mfenced></mrow>'
    assert mathml(x * Laplacian(ACS.x), printer='presentation') == '<mrow><mi>x</mi><mo>&InvisibleTimes;</mo><mfenced><mrow><mo>&#x2206;</mo><msub><mi mathvariant="bold">x</mi><mi mathvariant="bold">A</mi></msub></mrow></mfenced></mrow>'
    assert mathml(Laplacian(x * ACS.x), printer='presentation') == '<mrow><mo>&#x2206;</mo><mfenced><mrow><msub><mi mathvariant="bold">x</mi><mi mathvariant="bold">A</mi></msub><mo>&InvisibleTimes;</mo><mi>x</mi></mrow></mfenced></mrow>'