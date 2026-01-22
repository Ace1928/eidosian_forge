from sympy.concrete.products import Product
from sympy.core.function import Lambda
from sympy.core.numbers import (I, Rational, pi)
from sympy.core.singleton import S
from sympy.core.symbol import Dummy
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.integrals.integrals import Integral
from sympy.matrices.dense import Matrix
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.matrices.expressions.trace import Trace
from sympy.tensor.indexed import IndexedBase
from sympy.stats import (GaussianUnitaryEnsemble as GUE, density,
from sympy.stats.joint_rv_types import JointDistributionHandmade
from sympy.stats.rv import RandomMatrixSymbol
from sympy.stats.random_matrix_models import GaussianEnsemble, RandomMatrixPSpace
from sympy.testing.pytest import raises
def test_issue_19841():
    G1 = GUE('U', 2)
    G2 = G1.xreplace({2: 2})
    assert G1.args == G2.args
    X = MatrixSymbol('X', 2, 2)
    G = GSE('U', 2)
    h_pspace = RandomMatrixPSpace('P', model=density(G))
    H = RandomMatrixSymbol('H', 2, 2, pspace=h_pspace)
    H2 = RandomMatrixSymbol('H', 2, 2, pspace=None)
    assert H.doit() == H
    assert (2 * H).xreplace({H: X}) == 2 * X
    assert (2 * H).xreplace({H2: X}) == 2 * H
    assert (2 * H2).xreplace({H: X}) == 2 * H2
    assert (2 * H2).xreplace({H2: X}) == 2 * X