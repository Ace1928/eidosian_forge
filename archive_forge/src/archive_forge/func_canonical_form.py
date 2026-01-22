import random
import itertools
from typing import (Sequence as tSequence, Union as tUnion, List as tList,
from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.cache import cacheit
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.function import (Function, Lambda)
from sympy.core.mul import Mul
from sympy.core.numbers import (Integer, Rational, igcd, oo, pi)
from sympy.core.relational import (Eq, Ge, Gt, Le, Lt, Ne)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol)
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.integers import ceiling
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.special.gamma_functions import gamma
from sympy.logic.boolalg import (And, Not, Or)
from sympy.matrices.common import NonSquareMatrixError
from sympy.matrices.dense import (Matrix, eye, ones, zeros)
from sympy.matrices.expressions.blockmatrix import BlockMatrix
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.matrices.expressions.special import Identity
from sympy.matrices.immutable import ImmutableMatrix
from sympy.sets.conditionset import ConditionSet
from sympy.sets.contains import Contains
from sympy.sets.fancysets import Range
from sympy.sets.sets import (FiniteSet, Intersection, Interval, Set, Union)
from sympy.solvers.solveset import linsolve
from sympy.tensor.indexed import (Indexed, IndexedBase)
from sympy.core.relational import Relational
from sympy.logic.boolalg import Boolean
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.iterables import strongly_connected_components
from sympy.stats.joint_rv import JointDistribution
from sympy.stats.joint_rv_types import JointDistributionHandmade
from sympy.stats.rv import (RandomIndexedSymbol, random_symbols, RandomSymbol,
from sympy.stats.stochastic_process import StochasticPSpace
from sympy.stats.symbolic_probability import Probability, Expectation
from sympy.stats.frv_types import Bernoulli, BernoulliDistribution, FiniteRV
from sympy.stats.drv_types import Poisson, PoissonDistribution
from sympy.stats.crv_types import Normal, NormalDistribution, Gamma, GammaDistribution
from sympy.core.sympify import _sympify, sympify
def canonical_form(self) -> tTuple[tList[Basic], ImmutableMatrix]:
    """
        Reorders the one-step transition matrix
        so that recurrent states appear first and transient
        states appear last. Other representations include inserting
        transient states first and recurrent states last.

        Returns
        =======

        states, P_new
            ``states`` is the list that describes the order of the
            new states in the matrix
            so that the ith element in ``states`` is the state of the
            ith row of A.
            ``P_new`` is the new transition matrix in canonical form.

        Examples
        ========

        >>> from sympy.stats import DiscreteMarkovChain
        >>> from sympy import Matrix, S

        You can convert your chain into canonical form:

        >>> T = Matrix([[S(1)/2, S(1)/2, 0,      0,      0],
        ...             [S(2)/5, S(1)/5, S(2)/5, 0,      0],
        ...             [0,      0,      1,      0,      0],
        ...             [0,      0,      S(1)/2, S(1)/2, 0],
        ...             [S(1)/2, 0,      0,      0, S(1)/2]])
        >>> X = DiscreteMarkovChain('X', list(range(1, 6)), trans_probs=T)
        >>> states, new_matrix = X.canonical_form()
        >>> states
        [3, 1, 2, 4, 5]

        >>> new_matrix
        Matrix([
        [  1,   0,   0,   0,   0],
        [  0, 1/2, 1/2,   0,   0],
        [2/5, 2/5, 1/5,   0,   0],
        [1/2,   0,   0, 1/2,   0],
        [  0, 1/2,   0,   0, 1/2]])

        The new states are [3, 1, 2, 4, 5] and you can
        create a new chain with this and its canonical
        form will remain the same (since it is already
        in canonical form).

        >>> X = DiscreteMarkovChain('X', states, new_matrix)
        >>> states, new_matrix = X.canonical_form()
        >>> states
        [3, 1, 2, 4, 5]

        >>> new_matrix
        Matrix([
        [  1,   0,   0,   0,   0],
        [  0, 1/2, 1/2,   0,   0],
        [2/5, 2/5, 1/5,   0,   0],
        [1/2,   0,   0, 1/2,   0],
        [  0, 1/2,   0,   0, 1/2]])

        This is not limited to absorbing chains:

        >>> T = Matrix([[0, 5,  5, 0,  0],
        ...             [0, 0,  0, 10, 0],
        ...             [5, 0,  5, 0,  0],
        ...             [0, 10, 0, 0,  0],
        ...             [0, 3,  0, 3,  4]])/10
        >>> X = DiscreteMarkovChain('X', trans_probs=T)
        >>> states, new_matrix = X.canonical_form()
        >>> states
        [1, 3, 0, 2, 4]

        >>> new_matrix
        Matrix([
        [   0,    1,   0,   0,   0],
        [   1,    0,   0,   0,   0],
        [ 1/2,    0,   0, 1/2,   0],
        [   0,    0, 1/2, 1/2,   0],
        [3/10, 3/10,   0,   0, 2/5]])

        See Also
        ========

        sympy.stats.DiscreteMarkovChain.communication_classes
        sympy.stats.DiscreteMarkovChain.decompose

        References
        ==========

        .. [1] https://onlinelibrary.wiley.com/doi/pdf/10.1002/9780470316887.app1
        .. [2] http://www.columbia.edu/~ww2040/6711F12/lect1023big.pdf
        """
    states, A, B, C = self.decompose()
    O = zeros(A.shape[0], C.shape[1])
    return (states, BlockMatrix([[A, O], [B, C]]).as_explicit())