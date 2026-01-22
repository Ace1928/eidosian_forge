import multiprocessing
import platform
from scipy.optimize._differentialevolution import (DifferentialEvolutionSolver,
from scipy.optimize import differential_evolution, OptimizeResult
from scipy.optimize._constraints import (Bounds, NonlinearConstraint,
from scipy.optimize import rosen, minimize
from scipy.sparse import csr_matrix
from scipy import stats
import numpy as np
from numpy.testing import (assert_equal, assert_allclose, assert_almost_equal,
from pytest import raises as assert_raises, warns
import pytest
def test_strategy_fn(self):
    parameter_count = 4
    popsize = 10
    bounds = [(0, 10.0)] * parameter_count
    total_popsize = parameter_count * popsize
    mutation = 0.8
    recombination = 0.7

    def custom_strategy_fn(candidate, population, rng=None):
        trial = np.copy(population[candidate])
        fill_point = rng.choice(parameter_count)
        pool = np.arange(total_popsize)
        rng.shuffle(pool)
        idxs = []
        while len(idxs) < 2 and len(pool) > 0:
            idx = pool[0]
            pool = pool[1:]
            if idx != candidate:
                idxs.append(idx)
        r0, r1 = idxs[:2]
        bprime = population[0] + mutation * (population[r0] - population[r1])
        crossovers = rng.uniform(size=parameter_count)
        crossovers = crossovers < recombination
        crossovers[fill_point] = True
        trial = np.where(crossovers, bprime, trial)
        return trial
    solver = DifferentialEvolutionSolver(rosen, bounds, popsize=popsize, recombination=recombination, mutation=mutation, maxiter=2, strategy=custom_strategy_fn, seed=10, polish=False)
    assert solver.strategy is custom_strategy_fn
    res = solver.solve()
    res2 = differential_evolution(rosen, bounds, mutation=mutation, popsize=popsize, recombination=recombination, maxiter=2, strategy='best1bin', polish=False, seed=10)
    assert_allclose(res.population, res2.population)
    assert_allclose(res.x, res2.x)

    def custom_strategy_fn(candidate, population, rng=None):
        return np.array([1.0, 2.0])
    with pytest.raises(RuntimeError, match='strategy*'):
        differential_evolution(rosen, bounds, strategy=custom_strategy_fn)