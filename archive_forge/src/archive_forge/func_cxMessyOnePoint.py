import random
import warnings
from itertools import repeat
def cxMessyOnePoint(ind1, ind2):
    """Executes a one point crossover on :term:`sequence` individual.
    The crossover will in most cases change the individuals size. The two
    individuals are modified in place.

    :param ind1: The first individual participating in the crossover.
    :param ind2: The second individual participating in the crossover.
    :returns: A tuple of two individuals.

    This function uses the :func:`~random.randint` function from the python base
    :mod:`random` module.
    """
    cxpoint1 = random.randint(0, len(ind1))
    cxpoint2 = random.randint(0, len(ind2))
    ind1[cxpoint1:], ind2[cxpoint2:] = (ind2[cxpoint2:], ind1[cxpoint1:])
    return (ind1, ind2)