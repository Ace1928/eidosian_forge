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
def test_content_mathml_greek():
    mml = mp._print(Symbol('alpha'))
    assert mml.nodeName == 'ci'
    assert mml.childNodes[0].nodeValue == 'α'
    assert mp.doprint(Symbol('alpha')) == '<ci>&#945;</ci>'
    assert mp.doprint(Symbol('beta')) == '<ci>&#946;</ci>'
    assert mp.doprint(Symbol('gamma')) == '<ci>&#947;</ci>'
    assert mp.doprint(Symbol('delta')) == '<ci>&#948;</ci>'
    assert mp.doprint(Symbol('epsilon')) == '<ci>&#949;</ci>'
    assert mp.doprint(Symbol('zeta')) == '<ci>&#950;</ci>'
    assert mp.doprint(Symbol('eta')) == '<ci>&#951;</ci>'
    assert mp.doprint(Symbol('theta')) == '<ci>&#952;</ci>'
    assert mp.doprint(Symbol('iota')) == '<ci>&#953;</ci>'
    assert mp.doprint(Symbol('kappa')) == '<ci>&#954;</ci>'
    assert mp.doprint(Symbol('lambda')) == '<ci>&#955;</ci>'
    assert mp.doprint(Symbol('mu')) == '<ci>&#956;</ci>'
    assert mp.doprint(Symbol('nu')) == '<ci>&#957;</ci>'
    assert mp.doprint(Symbol('xi')) == '<ci>&#958;</ci>'
    assert mp.doprint(Symbol('omicron')) == '<ci>&#959;</ci>'
    assert mp.doprint(Symbol('pi')) == '<ci>&#960;</ci>'
    assert mp.doprint(Symbol('rho')) == '<ci>&#961;</ci>'
    assert mp.doprint(Symbol('varsigma')) == '<ci>&#962;</ci>'
    assert mp.doprint(Symbol('sigma')) == '<ci>&#963;</ci>'
    assert mp.doprint(Symbol('tau')) == '<ci>&#964;</ci>'
    assert mp.doprint(Symbol('upsilon')) == '<ci>&#965;</ci>'
    assert mp.doprint(Symbol('phi')) == '<ci>&#966;</ci>'
    assert mp.doprint(Symbol('chi')) == '<ci>&#967;</ci>'
    assert mp.doprint(Symbol('psi')) == '<ci>&#968;</ci>'
    assert mp.doprint(Symbol('omega')) == '<ci>&#969;</ci>'
    assert mp.doprint(Symbol('Alpha')) == '<ci>&#913;</ci>'
    assert mp.doprint(Symbol('Beta')) == '<ci>&#914;</ci>'
    assert mp.doprint(Symbol('Gamma')) == '<ci>&#915;</ci>'
    assert mp.doprint(Symbol('Delta')) == '<ci>&#916;</ci>'
    assert mp.doprint(Symbol('Epsilon')) == '<ci>&#917;</ci>'
    assert mp.doprint(Symbol('Zeta')) == '<ci>&#918;</ci>'
    assert mp.doprint(Symbol('Eta')) == '<ci>&#919;</ci>'
    assert mp.doprint(Symbol('Theta')) == '<ci>&#920;</ci>'
    assert mp.doprint(Symbol('Iota')) == '<ci>&#921;</ci>'
    assert mp.doprint(Symbol('Kappa')) == '<ci>&#922;</ci>'
    assert mp.doprint(Symbol('Lambda')) == '<ci>&#923;</ci>'
    assert mp.doprint(Symbol('Mu')) == '<ci>&#924;</ci>'
    assert mp.doprint(Symbol('Nu')) == '<ci>&#925;</ci>'
    assert mp.doprint(Symbol('Xi')) == '<ci>&#926;</ci>'
    assert mp.doprint(Symbol('Omicron')) == '<ci>&#927;</ci>'
    assert mp.doprint(Symbol('Pi')) == '<ci>&#928;</ci>'
    assert mp.doprint(Symbol('Rho')) == '<ci>&#929;</ci>'
    assert mp.doprint(Symbol('Sigma')) == '<ci>&#931;</ci>'
    assert mp.doprint(Symbol('Tau')) == '<ci>&#932;</ci>'
    assert mp.doprint(Symbol('Upsilon')) == '<ci>&#933;</ci>'
    assert mp.doprint(Symbol('Phi')) == '<ci>&#934;</ci>'
    assert mp.doprint(Symbol('Chi')) == '<ci>&#935;</ci>'
    assert mp.doprint(Symbol('Psi')) == '<ci>&#936;</ci>'
    assert mp.doprint(Symbol('Omega')) == '<ci>&#937;</ci>'