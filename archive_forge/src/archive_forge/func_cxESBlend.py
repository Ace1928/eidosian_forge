import random
import warnings
from itertools import repeat
def cxESBlend(ind1, ind2, alpha):
    """Executes a blend crossover on both, the individual and the strategy. The
    individuals shall be a :term:`sequence` and must have a :term:`sequence`
    :attr:`strategy` attribute. Adjustment of the minimal strategy shall be done
    after the call to this function, consider using a decorator.

    :param ind1: The first evolution strategy participating in the crossover.
    :param ind2: The second evolution strategy participating in the crossover.
    :param alpha: Extent of the interval in which the new values can be drawn
                  for each attribute on both side of the parents' attributes.
    :returns: A tuple of two evolution strategies.

    This function uses the :func:`~random.random` function from the python base
    :mod:`random` module.
    """
    for i, (x1, s1, x2, s2) in enumerate(zip(ind1, ind1.strategy, ind2, ind2.strategy)):
        gamma = (1.0 + 2.0 * alpha) * random.random() - alpha
        ind1[i] = (1.0 - gamma) * x1 + gamma * x2
        ind2[i] = gamma * x1 + (1.0 - gamma) * x2
        gamma = (1.0 + 2.0 * alpha) * random.random() - alpha
        ind1.strategy[i] = (1.0 - gamma) * s1 + gamma * s2
        ind2.strategy[i] = gamma * s1 + (1.0 - gamma) * s2
    return (ind1, ind2)