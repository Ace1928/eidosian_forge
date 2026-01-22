import numpy as np
from deap import tools, gp
from inspect import isclass
from .operator_utils import set_sample_weight
from sklearn.utils import indexable
from sklearn.metrics import check_scoring
from sklearn.model_selection._validation import _fit_and_score
from sklearn.base import clone
from collections import defaultdict
import warnings
from stopit import threading_timeoutable, TimeoutException
def pick_two_individuals_eligible_for_crossover(population):
    """Pick two individuals from the population which can do crossover, that is, they share a primitive.

    Parameters
    ----------
    population: array of individuals

    Returns
    ----------
    tuple: (individual, individual)
        Two individuals which are not the same, but share at least one primitive.
        Alternatively, if no such pair exists in the population, (None, None) is returned instead.
    """
    primitives_by_ind = [set([node.name for node in ind if isinstance(node, gp.Primitive)]) for ind in population]
    pop_as_str = [str(ind) for ind in population]
    eligible_pairs = [(i, i + 1 + j) for i, ind1_prims in enumerate(primitives_by_ind) for j, ind2_prims in enumerate(primitives_by_ind[i + 1:]) if not ind1_prims.isdisjoint(ind2_prims) and pop_as_str[i] != pop_as_str[i + 1 + j]]
    eligible_pairs += [(j, i) for i, j in eligible_pairs]
    if not eligible_pairs:
        return (None, None)
    pair = np.random.randint(0, len(eligible_pairs))
    idx1, idx2 = eligible_pairs[pair]
    return (population[idx1], population[idx2])