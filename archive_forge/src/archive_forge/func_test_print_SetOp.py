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
def test_print_SetOp():
    f1 = FiniteSet(x, 1, 3)
    f2 = FiniteSet(y, 2, 4)
    prntr = lambda x: mathml(x, printer='presentation')
    assert prntr(Union(f1, f2, evaluate=False)) == '<mrow><mfenced close="}" open="{"><mn>1</mn><mn>3</mn><mi>x</mi></mfenced><mo>&#x222A;</mo><mfenced close="}" open="{"><mn>2</mn><mn>4</mn><mi>y</mi></mfenced></mrow>'
    assert prntr(Intersection(f1, f2, evaluate=False)) == '<mrow><mfenced close="}" open="{"><mn>1</mn><mn>3</mn><mi>x</mi></mfenced><mo>&#x2229;</mo><mfenced close="}" open="{"><mn>2</mn><mn>4</mn><mi>y</mi></mfenced></mrow>'
    assert prntr(Complement(f1, f2, evaluate=False)) == '<mrow><mfenced close="}" open="{"><mn>1</mn><mn>3</mn><mi>x</mi></mfenced><mo>&#x2216;</mo><mfenced close="}" open="{"><mn>2</mn><mn>4</mn><mi>y</mi></mfenced></mrow>'
    assert prntr(SymmetricDifference(f1, f2, evaluate=False)) == '<mrow><mfenced close="}" open="{"><mn>1</mn><mn>3</mn><mi>x</mi></mfenced><mo>&#x2206;</mo><mfenced close="}" open="{"><mn>2</mn><mn>4</mn><mi>y</mi></mfenced></mrow>'
    A = FiniteSet(a)
    C = FiniteSet(c)
    D = FiniteSet(d)
    U1 = Union(C, D, evaluate=False)
    I1 = Intersection(C, D, evaluate=False)
    C1 = Complement(C, D, evaluate=False)
    D1 = SymmetricDifference(C, D, evaluate=False)
    P1 = ProductSet(C, D)
    assert prntr(Union(A, I1, evaluate=False)) == '<mrow><mfenced close="}" open="{"><mi>a</mi></mfenced><mo>&#x222A;</mo><mfenced><mrow><mfenced close="}" open="{"><mi>c</mi></mfenced><mo>&#x2229;</mo><mfenced close="}" open="{"><mi>d</mi></mfenced></mrow></mfenced></mrow>'
    assert prntr(Intersection(A, C1, evaluate=False)) == '<mrow><mfenced close="}" open="{"><mi>a</mi></mfenced><mo>&#x2229;</mo><mfenced><mrow><mfenced close="}" open="{"><mi>c</mi></mfenced><mo>&#x2216;</mo><mfenced close="}" open="{"><mi>d</mi></mfenced></mrow></mfenced></mrow>'
    assert prntr(Complement(A, D1, evaluate=False)) == '<mrow><mfenced close="}" open="{"><mi>a</mi></mfenced><mo>&#x2216;</mo><mfenced><mrow><mfenced close="}" open="{"><mi>c</mi></mfenced><mo>&#x2206;</mo><mfenced close="}" open="{"><mi>d</mi></mfenced></mrow></mfenced></mrow>'
    assert prntr(SymmetricDifference(A, P1, evaluate=False)) == '<mrow><mfenced close="}" open="{"><mi>a</mi></mfenced><mo>&#x2206;</mo><mfenced><mrow><mfenced close="}" open="{"><mi>c</mi></mfenced><mo>&#x00d7;</mo><mfenced close="}" open="{"><mi>d</mi></mfenced></mrow></mfenced></mrow>'
    assert prntr(ProductSet(A, U1)) == '<mrow><mfenced close="}" open="{"><mi>a</mi></mfenced><mo>&#x00d7;</mo><mfenced><mrow><mfenced close="}" open="{"><mi>c</mi></mfenced><mo>&#x222A;</mo><mfenced close="}" open="{"><mi>d</mi></mfenced></mrow></mfenced></mrow>'