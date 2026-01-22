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
def communication_classes(self) -> tList[tTuple[tList[Basic], Boolean, Integer]]:
    """
        Returns the list of communication classes that partition
        the states of the markov chain.

        A communication class is defined to be a set of states
        such that every state in that set is reachable from
        every other state in that set. Due to its properties
        this forms a class in the mathematical sense.
        Communication classes are also known as recurrence
        classes.

        Returns
        =======

        classes
            The ``classes`` are a list of tuples. Each
            tuple represents a single communication class
            with its properties. The first element in the
            tuple is the list of states in the class, the
            second element is whether the class is recurrent
            and the third element is the period of the
            communication class.

        Examples
        ========

        >>> from sympy.stats import DiscreteMarkovChain
        >>> from sympy import Matrix
        >>> T = Matrix([[0, 1, 0],
        ...             [1, 0, 0],
        ...             [1, 0, 0]])
        >>> X = DiscreteMarkovChain('X', [1, 2, 3], T)
        >>> classes = X.communication_classes()
        >>> for states, is_recurrent, period in classes:
        ...     states, is_recurrent, period
        ([1, 2], True, 2)
        ([3], False, 1)

        From this we can see that states ``1`` and ``2``
        communicate, are recurrent and have a period
        of 2. We can also see state ``3`` is transient
        with a period of 1.

        Notes
        =====

        The algorithm used is of order ``O(n**2)`` where
        ``n`` is the number of states in the markov chain.
        It uses Tarjan's algorithm to find the classes
        themselves and then it uses a breadth-first search
        algorithm to find each class's periodicity.
        Most of the algorithm's components approach ``O(n)``
        as the matrix becomes more and more sparse.

        References
        ==========

        .. [1] https://web.archive.org/web/20220207032113/https://www.columbia.edu/~ww2040/4701Sum07/4701-06-Notes-MCII.pdf
        .. [2] https://cecas.clemson.edu/~shierd/Shier/markov.pdf
        .. [3] https://ujcontent.uj.ac.za/esploro/outputs/graduate/Markov-chains--a-graph-theoretical/999849107691#file-0
        .. [4] https://www.mathworks.com/help/econ/dtmc.classify.html
        """
    n = self.number_of_states
    T = self.transition_probabilities
    if isinstance(T, MatrixSymbol):
        raise NotImplementedError('Cannot perform the operation with a symbolic matrix.')
    V = Range(n)
    E = [(i, j) for i in V for j in V if T[i, j] != 0]
    classes = strongly_connected_components((V, E))
    recurrence = []
    periods = []
    for class_ in classes:
        submatrix = T[class_, class_]
        is_recurrent = S.true
        rows = submatrix.tolist()
        for row in rows:
            if sum(row) - 1 != 0:
                is_recurrent = S.false
                break
        recurrence.append(is_recurrent)
        non_tree_edge_values: tSet[int] = set()
        visited = {class_[0]}
        newly_visited = {class_[0]}
        level = {class_[0]: 0}
        current_level = 0
        done = False
        while not done:
            done = len(visited) == len(class_)
            current_level += 1
            for i in newly_visited:
                newly_visited = {j for j in class_ if T[i, j] != 0}
                new_tree_edges = newly_visited.difference(visited)
                for j in new_tree_edges:
                    level[j] = current_level
                new_non_tree_edges = newly_visited.intersection(visited)
                new_non_tree_edge_values = {level[i] - level[j] + 1 for j in new_non_tree_edges}
                non_tree_edge_values = non_tree_edge_values.union(new_non_tree_edge_values)
                visited = visited.union(new_tree_edges)
        positive_ntev = {val_e for val_e in non_tree_edge_values if val_e > 0}
        if len(positive_ntev) == 0:
            periods.append(len(class_))
        elif len(positive_ntev) == 1:
            periods.append(positive_ntev.pop())
        else:
            periods.append(igcd(*positive_ntev))
    classes = [[_sympify(self._state_index[i]) for i in class_] for class_ in classes]
    return list(zip(classes, recurrence, map(Integer, periods)))