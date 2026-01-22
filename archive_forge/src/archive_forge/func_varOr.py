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
def varOr(population, toolbox, lambda_, cxpb, mutpb):
    """Part of an evolutionary algorithm applying only the variation part
    (crossover, mutation **or** reproduction). The modified individuals have
    their fitness invalidated. The individuals are cloned so returned
    population is independent of the input population.
    :param population: A list of individuals to vary.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param lambda\\_: The number of children to produce
    :param cxpb: The probability of mating two individuals.
    :param mutpb: The probability of mutating an individual.
    :returns: The final population
    :returns: A class:`~deap.tools.Logbook` with the statistics of the
              evolution
    The variation goes as follow. On each of the *lambda_* iteration, it
    selects one of the three operations; crossover, mutation or reproduction.
    In the case of a crossover, two individuals are selected at random from
    the parental population :math:`P_\\mathrm{p}`, those individuals are cloned
    using the :meth:`toolbox.clone` method and then mated using the
    :meth:`toolbox.mate` method. Only the first child is appended to the
    offspring population :math:`P_\\mathrm{o}`, the second child is discarded.
    In the case of a mutation, one individual is selected at random from
    :math:`P_\\mathrm{p}`, it is cloned and then mutated using using the
    :meth:`toolbox.mutate` method. The resulting mutant is appended to
    :math:`P_\\mathrm{o}`. In the case of a reproduction, one individual is
    selected at random from :math:`P_\\mathrm{p}`, cloned and appended to
    :math:`P_\\mathrm{o}`.
    This variation is named *Or* beceause an offspring will never result from
    both operations crossover and mutation. The sum of both probabilities
    shall be in :math:`[0, 1]`, the reproduction probability is
    1 - *cxpb* - *mutpb*.
    """
    offspring = []
    for _ in range(lambda_):
        op_choice = np.random.random()
        if op_choice < cxpb:
            ind1, ind2 = pick_two_individuals_eligible_for_crossover(population)
            if ind1 is not None:
                ind1_cx, _, evaluated_individuals_ = toolbox.mate(ind1, ind2)
                del ind1_cx.fitness.values
                if str(ind1_cx) in evaluated_individuals_:
                    ind1_cx = mutate_random_individual(population, toolbox)
                offspring.append(ind1_cx)
            else:
                ind_mu = mutate_random_individual(population, toolbox)
                offspring.append(ind_mu)
        elif op_choice < cxpb + mutpb:
            ind = mutate_random_individual(population, toolbox)
            offspring.append(ind)
        else:
            idx = np.random.randint(0, len(population))
            offspring.append(toolbox.clone(population[idx]))
    return offspring