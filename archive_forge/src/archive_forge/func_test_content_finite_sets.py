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
def test_content_finite_sets():
    assert mathml(FiniteSet(a)) == '<set><ci>a</ci></set>'
    assert mathml(FiniteSet(a, b)) == '<set><ci>a</ci><ci>b</ci></set>'
    assert mathml(FiniteSet(FiniteSet(a, b), c)) == '<set><ci>c</ci><set><ci>a</ci><ci>b</ci></set></set>'
    A = FiniteSet(a)
    B = FiniteSet(b)
    C = FiniteSet(c)
    D = FiniteSet(d)
    U1 = Union(A, B, evaluate=False)
    U2 = Union(C, D, evaluate=False)
    I1 = Intersection(A, B, evaluate=False)
    I2 = Intersection(C, D, evaluate=False)
    C1 = Complement(A, B, evaluate=False)
    C2 = Complement(C, D, evaluate=False)
    P1 = ProductSet(A, B)
    P2 = ProductSet(C, D)
    assert mathml(U1) == '<apply><union/><set><ci>a</ci></set><set><ci>b</ci></set></apply>'
    assert mathml(I1) == '<apply><intersect/><set><ci>a</ci></set><set><ci>b</ci></set></apply>'
    assert mathml(C1) == '<apply><setdiff/><set><ci>a</ci></set><set><ci>b</ci></set></apply>'
    assert mathml(P1) == '<apply><cartesianproduct/><set><ci>a</ci></set><set><ci>b</ci></set></apply>'
    assert mathml(Intersection(A, U2, evaluate=False)) == '<apply><intersect/><set><ci>a</ci></set><apply><union/><set><ci>c</ci></set><set><ci>d</ci></set></apply></apply>'
    assert mathml(Intersection(U1, U2, evaluate=False)) == '<apply><intersect/><apply><union/><set><ci>a</ci></set><set><ci>b</ci></set></apply><apply><union/><set><ci>c</ci></set><set><ci>d</ci></set></apply></apply>'
    assert mathml(Intersection(C1, C2, evaluate=False)) == '<apply><intersect/><apply><setdiff/><set><ci>a</ci></set><set><ci>b</ci></set></apply><apply><setdiff/><set><ci>c</ci></set><set><ci>d</ci></set></apply></apply>'
    assert mathml(Intersection(P1, P2, evaluate=False)) == '<apply><intersect/><apply><cartesianproduct/><set><ci>a</ci></set><set><ci>b</ci></set></apply><apply><cartesianproduct/><set><ci>c</ci></set><set><ci>d</ci></set></apply></apply>'
    assert mathml(Union(A, I2, evaluate=False)) == '<apply><union/><set><ci>a</ci></set><apply><intersect/><set><ci>c</ci></set><set><ci>d</ci></set></apply></apply>'
    assert mathml(Union(I1, I2, evaluate=False)) == '<apply><union/><apply><intersect/><set><ci>a</ci></set><set><ci>b</ci></set></apply><apply><intersect/><set><ci>c</ci></set><set><ci>d</ci></set></apply></apply>'
    assert mathml(Union(C1, C2, evaluate=False)) == '<apply><union/><apply><setdiff/><set><ci>a</ci></set><set><ci>b</ci></set></apply><apply><setdiff/><set><ci>c</ci></set><set><ci>d</ci></set></apply></apply>'
    assert mathml(Union(P1, P2, evaluate=False)) == '<apply><union/><apply><cartesianproduct/><set><ci>a</ci></set><set><ci>b</ci></set></apply><apply><cartesianproduct/><set><ci>c</ci></set><set><ci>d</ci></set></apply></apply>'
    assert mathml(Complement(A, C2, evaluate=False)) == '<apply><setdiff/><set><ci>a</ci></set><apply><setdiff/><set><ci>c</ci></set><set><ci>d</ci></set></apply></apply>'
    assert mathml(Complement(U1, U2, evaluate=False)) == '<apply><setdiff/><apply><union/><set><ci>a</ci></set><set><ci>b</ci></set></apply><apply><union/><set><ci>c</ci></set><set><ci>d</ci></set></apply></apply>'
    assert mathml(Complement(I1, I2, evaluate=False)) == '<apply><setdiff/><apply><intersect/><set><ci>a</ci></set><set><ci>b</ci></set></apply><apply><intersect/><set><ci>c</ci></set><set><ci>d</ci></set></apply></apply>'
    assert mathml(Complement(P1, P2, evaluate=False)) == '<apply><setdiff/><apply><cartesianproduct/><set><ci>a</ci></set><set><ci>b</ci></set></apply><apply><cartesianproduct/><set><ci>c</ci></set><set><ci>d</ci></set></apply></apply>'
    assert mathml(ProductSet(A, P2)) == '<apply><cartesianproduct/><set><ci>a</ci></set><apply><cartesianproduct/><set><ci>c</ci></set><set><ci>d</ci></set></apply></apply>'
    assert mathml(ProductSet(U1, U2)) == '<apply><cartesianproduct/><apply><union/><set><ci>a</ci></set><set><ci>b</ci></set></apply><apply><union/><set><ci>c</ci></set><set><ci>d</ci></set></apply></apply>'
    assert mathml(ProductSet(I1, I2)) == '<apply><cartesianproduct/><apply><intersect/><set><ci>a</ci></set><set><ci>b</ci></set></apply><apply><intersect/><set><ci>c</ci></set><set><ci>d</ci></set></apply></apply>'
    assert mathml(ProductSet(C1, C2)) == '<apply><cartesianproduct/><apply><setdiff/><set><ci>a</ci></set><set><ci>b</ci></set></apply><apply><setdiff/><set><ci>c</ci></set><set><ci>d</ci></set></apply></apply>'