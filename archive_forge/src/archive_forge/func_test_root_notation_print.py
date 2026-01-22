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
def test_root_notation_print():
    assert mathml(x ** (S.One / 3), printer='presentation') == '<mroot><mi>x</mi><mn>3</mn></mroot>'
    assert mathml(x ** (S.One / 3), printer='presentation', root_notation=False) == '<msup><mi>x</mi><mfrac><mn>1</mn><mn>3</mn></mfrac></msup>'
    assert mathml(x ** (S.One / 3), printer='content') == '<apply><root/><degree><cn>3</cn></degree><ci>x</ci></apply>'
    assert mathml(x ** (S.One / 3), printer='content', root_notation=False) == '<apply><power/><ci>x</ci><apply><divide/><cn>1</cn><cn>3</cn></apply></apply>'
    assert mathml(x ** Rational(-1, 3), printer='presentation') == '<mfrac><mn>1</mn><mroot><mi>x</mi><mn>3</mn></mroot></mfrac>'
    assert mathml(x ** Rational(-1, 3), printer='presentation', root_notation=False) == '<mfrac><mn>1</mn><msup><mi>x</mi><mfrac><mn>1</mn><mn>3</mn></mfrac></msup></mfrac>'