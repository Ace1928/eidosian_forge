import random
import warnings
from itertools import repeat
def cxTwoPoint(ind1, ind2):
    """Executes a two-point crossover on the input :term:`sequence`
    individuals. The two individuals are modified in place and both keep
    their original length.

    :param ind1: The first individual participating in the crossover.
    :param ind2: The second individual participating in the crossover.
    :returns: A tuple of two individuals.

    This function uses the :func:`~random.randint` function from the Python
    base :mod:`random` module.
    """
    size = min(len(ind1), len(ind2))
    cxpoint1 = random.randint(1, size)
    cxpoint2 = random.randint(1, size - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else:
        cxpoint1, cxpoint2 = (cxpoint2, cxpoint1)
    ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] = (ind2[cxpoint1:cxpoint2], ind1[cxpoint1:cxpoint2])
    return (ind1, ind2)