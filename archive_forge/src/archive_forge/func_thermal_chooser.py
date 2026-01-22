import functools
import heapq
import math
import numbers
import time
from collections import deque
from . import helpers, paths
def thermal_chooser(queue, remaining, nbranch=8, temperature=1, rel_temperature=True):
    """A contraction 'chooser' that weights possible contractions using a
    Boltzmann distribution. Explicitly, given costs ``c_i`` (with ``c_0`` the
    smallest), the relative weights, ``w_i``, are computed as:

        w_i = exp( -(c_i - c_0) / temperature)

    Additionally, if ``rel_temperature`` is set, scale ``temperature`` by
    ``abs(c_0)`` to account for likely fluctuating cost magnitudes during the
    course of a contraction.

    Parameters
    ----------
    queue : list
        The heapified list of candidate contractions.
    remaining : dict[str, int]
        Mapping of remaining inputs' indices to the ssa id.
    temperature : float, optional
        When choosing a possible contraction, its relative probability will be
        proportional to ``exp(-cost / temperature)``. Thus the larger
        ``temperature`` is, the further random paths will stray from the normal
        'greedy' path. Conversely, if set to zero, only paths with exactly the
        same cost as the best at each step will be explored.
    rel_temperature : bool, optional
        Whether to normalize the ``temperature`` at each step to the scale of
        the best cost. This is generally beneficial as the magnitude of costs
        can vary significantly throughout a contraction.
    nbranch : int, optional
        How many potential paths to calculate probability for and choose from
        at each step.

    Returns
    -------
    cost, k1, k2, k12
    """
    n = 0
    choices = []
    while queue and n < nbranch:
        cost, k1, k2, k12 = heapq.heappop(queue)
        if k1 not in remaining or k2 not in remaining:
            continue
        choices.append((cost, k1, k2, k12))
        n += 1
    if n == 0:
        return None
    if n == 1:
        return choices[0]
    costs = [choice[0][0] for choice in choices]
    cmin = costs[0]
    if rel_temperature:
        temperature *= max(1, abs(cmin))
    if temperature == 0.0:
        energies = [1 if c == cmin else 0 for c in costs]
    else:
        energies = [math.exp(-(c - cmin) / temperature) for c in costs]
    chosen, = random_choices(range(n), weights=energies)
    cost, k1, k2, k12 = choices.pop(chosen)
    for other in choices:
        heapq.heappush(queue, other)
    return (cost, k1, k2, k12)