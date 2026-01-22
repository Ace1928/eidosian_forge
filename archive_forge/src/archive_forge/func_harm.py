import copy
import math
import copyreg
import random
import re
import sys
import types
import warnings
from collections import defaultdict, deque
from functools import partial, wraps
from operator import eq, lt
from . import tools  # Needed by HARM-GP
def harm(population, toolbox, cxpb, mutpb, ngen, alpha, beta, gamma, rho, nbrindsmodel=-1, mincutoff=20, stats=None, halloffame=None, verbose=__debug__):
    """Implement bloat control on a GP evolution using HARM-GP, as defined in
    [Gardner2015]. It is implemented in the form of an evolution algorithm
    (similar to :func:`~deap.algorithms.eaSimple`).

    :param population: A list of individuals.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param cxpb: The probability of mating two individuals.
    :param mutpb: The probability of mutating an individual.
    :param ngen: The number of generation.
    :param alpha: The HARM *alpha* parameter.
    :param beta: The HARM *beta* parameter.
    :param gamma: The HARM *gamma* parameter.
    :param rho: The HARM *rho* parameter.
    :param nbrindsmodel: The number of individuals to generate in order to
                            model the natural distribution. -1 is a special
                            value which uses the equation proposed in
                            [Gardner2015] to set the value of this parameter :
                            max(2000, len(population))
    :param mincutoff: The absolute minimum value for the cutoff point. It is
                        used to ensure that HARM does not shrink the population
                        too much at the beginning of the evolution. The default
                        value is usually fine.
    :param stats: A :class:`~deap.tools.Statistics` object that is updated
                  inplace, optional.
    :param halloffame: A :class:`~deap.tools.HallOfFame` object that will
                       contain the best individuals, optional.
    :param verbose: Whether or not to log the statistics.
    :returns: The final population
    :returns: A class:`~deap.tools.Logbook` with the statistics of the
              evolution

    This function expects the :meth:`toolbox.mate`, :meth:`toolbox.mutate`,
    :meth:`toolbox.select` and :meth:`toolbox.evaluate` aliases to be
    registered in the toolbox.

    .. note::
       The recommended values for the HARM-GP parameters are *alpha=0.05*,
       *beta=10*, *gamma=0.25*, *rho=0.9*. However, these parameters can be
       adjusted to perform better on a specific problem (see the relevant
       paper for tuning information). The number of individuals used to
       model the natural distribution and the minimum cutoff point are less
       important, their default value being effective in most cases.

    .. [Gardner2015] M.-A. Gardner, C. Gagne, and M. Parizeau, Controlling
        Code Growth by Dynamically Shaping the Genotype Size Distribution,
        Genetic Programming and Evolvable Machines, 2015,
        DOI 10.1007/s10710-015-9242-8

    """

    def _genpop(n, pickfrom=[], acceptfunc=lambda s: True, producesizes=False):
        producedpop = []
        producedpopsizes = []
        while len(producedpop) < n:
            if len(pickfrom) > 0:
                aspirant = pickfrom.pop()
                if acceptfunc(len(aspirant)):
                    producedpop.append(aspirant)
                    if producesizes:
                        producedpopsizes.append(len(aspirant))
            else:
                opRandom = random.random()
                if opRandom < cxpb:
                    aspirant1, aspirant2 = toolbox.mate(*map(toolbox.clone, toolbox.select(population, 2)))
                    del aspirant1.fitness.values, aspirant2.fitness.values
                    if acceptfunc(len(aspirant1)):
                        producedpop.append(aspirant1)
                        if producesizes:
                            producedpopsizes.append(len(aspirant1))
                    if len(producedpop) < n and acceptfunc(len(aspirant2)):
                        producedpop.append(aspirant2)
                        if producesizes:
                            producedpopsizes.append(len(aspirant2))
                else:
                    aspirant = toolbox.clone(toolbox.select(population, 1)[0])
                    if opRandom - cxpb < mutpb:
                        aspirant = toolbox.mutate(aspirant)[0]
                        del aspirant.fitness.values
                    if acceptfunc(len(aspirant)):
                        producedpop.append(aspirant)
                        if producesizes:
                            producedpopsizes.append(len(aspirant))
        if producesizes:
            return (producedpop, producedpopsizes)
        else:
            return producedpop

    def halflifefunc(x):
        return x * float(alpha) + beta
    if nbrindsmodel == -1:
        nbrindsmodel = max(2000, len(population))
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    if halloffame is not None:
        halloffame.update(population)
    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)
    for gen in range(1, ngen + 1):
        naturalpop, naturalpopsizes = _genpop(nbrindsmodel, producesizes=True)
        naturalhist = [0] * (max(naturalpopsizes) + 3)
        for indsize in naturalpopsizes:
            naturalhist[indsize] += 0.4
            naturalhist[indsize - 1] += 0.2
            naturalhist[indsize + 1] += 0.2
            naturalhist[indsize + 2] += 0.1
            if indsize - 2 >= 0:
                naturalhist[indsize - 2] += 0.1
        naturalhist = [val * len(population) / nbrindsmodel for val in naturalhist]
        sortednatural = sorted(naturalpop, key=lambda ind: ind.fitness)
        cutoffcandidates = sortednatural[int(len(population) * rho - 1):]
        cutoffsize = max(mincutoff, len(min(cutoffcandidates, key=len)))

        def targetfunc(x):
            return gamma * len(population) * math.log(2) / halflifefunc(x) * math.exp(-math.log(2) * (x - cutoffsize) / halflifefunc(x))
        targethist = [naturalhist[binidx] if binidx <= cutoffsize else targetfunc(binidx) for binidx in range(len(naturalhist))]
        probhist = [t / n if n > 0 else t for n, t in zip(naturalhist, targethist)]

        def probfunc(s):
            return probhist[s] if s < len(probhist) else targetfunc(s)

        def acceptfunc(s):
            return random.random() <= probfunc(s)
        offspring = _genpop(len(population), pickfrom=naturalpop, acceptfunc=acceptfunc, producesizes=False)
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        if halloffame is not None:
            halloffame.update(offspring)
        population[:] = offspring
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)
    return (population, logbook)