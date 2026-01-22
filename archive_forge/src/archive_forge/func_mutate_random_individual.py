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
def mutate_random_individual(population, toolbox):
    """Picks a random individual from the population, and performs mutation on a copy of it.

    Parameters
    ----------
    population: array of individuals

    Returns
    ----------
    individual: individual
        An individual which is a mutated copy of one of the individuals in population,
        the returned individual does not have fitness.values
    """
    idx = np.random.randint(0, len(population))
    ind = population[idx]
    ind, = toolbox.mutate(ind)
    del ind.fitness.values
    return ind