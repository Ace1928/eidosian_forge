import bisect
from collections import defaultdict, namedtuple
from itertools import chain
import math
from operator import attrgetter, itemgetter
import random
import numpy
def selTournamentDCD(individuals, k):
    """Tournament selection based on dominance (D) between two individuals, if
    the two individuals do not interdominate the selection is made
    based on crowding distance (CD). The *individuals* sequence length has to
    be a multiple of 4 only if k is equal to the length of individuals.
    Starting from the beginning of the selected individuals, two consecutive
    individuals will be different (assuming all individuals in the input list
    are unique). Each individual from the input list won't be selected more
    than twice.

    This selection requires the individuals to have a :attr:`crowding_dist`
    attribute, which can be set by the :func:`assignCrowdingDist` function.

    :param individuals: A list of individuals to select from.
    :param k: The number of individuals to select. Must be less than or equal
              to len(individuals).
    :returns: A list of selected individuals.
    """
    if k > len(individuals):
        raise ValueError('selTournamentDCD: k must be less than or equal to individuals length')
    if k == len(individuals) and k % 4 != 0:
        raise ValueError('selTournamentDCD: k must be divisible by four if k == len(individuals)')

    def tourn(ind1, ind2):
        if ind1.fitness.dominates(ind2.fitness):
            return ind1
        elif ind2.fitness.dominates(ind1.fitness):
            return ind2
        if ind1.fitness.crowding_dist < ind2.fitness.crowding_dist:
            return ind2
        elif ind1.fitness.crowding_dist > ind2.fitness.crowding_dist:
            return ind1
        if random.random() <= 0.5:
            return ind1
        return ind2
    individuals_1 = random.sample(individuals, len(individuals))
    individuals_2 = random.sample(individuals, len(individuals))
    chosen = []
    for i in range(0, k, 4):
        chosen.append(tourn(individuals_1[i], individuals_1[i + 1]))
        chosen.append(tourn(individuals_1[i + 2], individuals_1[i + 3]))
        chosen.append(tourn(individuals_2[i], individuals_2[i + 1]))
        chosen.append(tourn(individuals_2[i + 2], individuals_2[i + 3]))
    return chosen