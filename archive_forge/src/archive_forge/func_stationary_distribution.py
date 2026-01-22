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
def stationary_distribution(self, condition_set=False) -> tUnion[ImmutableMatrix, ConditionSet, Lambda]:
    """
        The stationary distribution is any row vector, p, that solves p = pP,
        is row stochastic and each element in p must be nonnegative.
        That means in matrix form: :math:`(P-I)^T p^T = 0` and
        :math:`(1, \\dots, 1) p = 1`
        where ``P`` is the one-step transition matrix.

        All time-homogeneous Markov Chains with a finite state space
        have at least one stationary distribution. In addition, if
        a finite time-homogeneous Markov Chain is irreducible, the
        stationary distribution is unique.

        Parameters
        ==========

        condition_set : bool
            If the chain has a symbolic size or transition matrix,
            it will return a ``Lambda`` if ``False`` and return a
            ``ConditionSet`` if ``True``.

        Examples
        ========

        >>> from sympy.stats import DiscreteMarkovChain
        >>> from sympy import Matrix, S

        An irreducible Markov Chain

        >>> T = Matrix([[S(1)/2, S(1)/2, 0],
        ...             [S(4)/5, S(1)/5, 0],
        ...             [1, 0, 0]])
        >>> X = DiscreteMarkovChain('X', trans_probs=T)
        >>> X.stationary_distribution()
        Matrix([[8/13, 5/13, 0]])

        A reducible Markov Chain

        >>> T = Matrix([[S(1)/2, S(1)/2, 0],
        ...             [S(4)/5, S(1)/5, 0],
        ...             [0, 0, 1]])
        >>> X = DiscreteMarkovChain('X', trans_probs=T)
        >>> X.stationary_distribution()
        Matrix([[8/13 - 8*tau0/13, 5/13 - 5*tau0/13, tau0]])

        >>> Y = DiscreteMarkovChain('Y')
        >>> Y.stationary_distribution()
        Lambda((wm, _T), Eq(wm*_T, wm))

        >>> Y.stationary_distribution(condition_set=True)
        ConditionSet(wm, Eq(wm*_T, wm))

        References
        ==========

        .. [1] https://www.probabilitycourse.com/chapter11/11_2_6_stationary_and_limiting_distributions.php
        .. [2] https://web.archive.org/web/20210508104430/https://galton.uchicago.edu/~yibi/teaching/stat317/2014/Lectures/Lecture4_6up.pdf

        See Also
        ========

        sympy.stats.DiscreteMarkovChain.limiting_distribution
        """
    trans_probs = self.transition_probabilities
    n = self.number_of_states
    if n == 0:
        return ImmutableMatrix(Matrix([[]]))
    if isinstance(trans_probs, MatrixSymbol):
        wm = MatrixSymbol('wm', 1, n)
        if condition_set:
            return ConditionSet(wm, Eq(wm * trans_probs, wm))
        else:
            return Lambda((wm, trans_probs), Eq(wm * trans_probs, wm))
    a = Matrix(trans_probs - Identity(n)).T
    a[0, 0:n] = ones(1, n)
    b = zeros(n, 1)
    b[0, 0] = 1
    soln = list(linsolve((a, b)))[0]
    return ImmutableMatrix([soln])