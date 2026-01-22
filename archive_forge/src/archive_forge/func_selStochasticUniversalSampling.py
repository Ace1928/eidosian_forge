import random
import numpy as np
from functools import partial
from operator import attrgetter
def selStochasticUniversalSampling(individuals, k, fit_attr='fitness'):
    """Select the *k* individuals among the input *individuals*.
    The selection is made by using a single random value to sample all of the
    individuals by choosing them at evenly spaced intervals. The list returned
    contains references to the input *individuals*.

    :param individuals: A list of individuals to select from.
    :param k: The number of individuals to select.
    :param fit_attr: The attribute of individuals to use as selection criterion
    :return: A list of selected individuals.

    This function uses the :func:`~random.uniform` function from the python base
    :mod:`random` module.
    """
    s_inds = sorted(individuals, key=attrgetter(fit_attr), reverse=True)
    sum_fits = sum((getattr(ind, fit_attr).values[0] for ind in individuals))
    distance = sum_fits / float(k)
    start = random.uniform(0, distance)
    points = [start + i * distance for i in range(k)]
    chosen = []
    for p in points:
        i = 0
        sum_ = getattr(s_inds[i], fit_attr).values[0]
        while sum_ < p:
            i += 1
            sum_ += getattr(s_inds[i], fit_attr).values[0]
        chosen.append(s_inds[i])
    return chosen