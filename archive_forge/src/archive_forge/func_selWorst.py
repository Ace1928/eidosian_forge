import random
import numpy as np
from functools import partial
from operator import attrgetter
def selWorst(individuals, k, fit_attr='fitness'):
    """Select the *k* worst individuals among the input *individuals*. The
    list returned contains references to the input *individuals*.

    :param individuals: A list of individuals to select from.
    :param k: The number of individuals to select.
    :param fit_attr: The attribute of individuals to use as selection criterion
    :returns: A list containing the k worst individuals.
    """
    return sorted(individuals, key=attrgetter(fit_attr))[:k]