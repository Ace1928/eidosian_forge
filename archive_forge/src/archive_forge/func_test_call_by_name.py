import collections
import functools
import math
import multiprocessing
import os
import random
import subprocess
import sys
import threading
import itertools
from textwrap import dedent
import numpy as np
import unittest
import numba
from numba import jit, _helperlib, njit
from numba.core import types
from numba.tests.support import TestCase, compile_function, tag
from numba.core.errors import TypingError
def test_call_by_name(self):
    data = {'np.random.beta': {'a': 1.0, 'b': 2.0, 'size': 3}, 'np.random.binomial': {'n': 1, 'p': 0.3, 'size': 3}, 'np.random.chisquare': {'df': 2.0, 'size': 3}, 'np.random.choice': {'a': 2, 'size': 3}, 'np.random.dirichlet': {'alpha': (2,), 'size': 3}, 'np.random.exponential': {'scale': 1.0, 'size': 3}, 'np.random.f': {'dfnum': 1.0, 'dfden': 2.0, 'size': 3}, 'np.random.gamma': {'shape': 2, 'scale': 2.0, 'size': 3}, 'np.random.geometric': {'p': 1.0, 'size': 3}, 'np.random.gumbel': {'loc': 0.0, 'scale': 1.0, 'size': 3}, 'np.random.hypergeometric': {'ngood': 1, 'nbad': 1, 'nsample': 1, 'size': 3}, 'np.random.laplace': {'loc': 0.0, 'scale': 1.0, 'size': 3}, 'np.random.logistic': {'loc': 0.0, 'scale': 1.0, 'size': 3}, 'np.random.lognormal': {'mean': 0.0, 'sigma': 1.0, 'size': 3}, 'np.random.logseries': {'p': 0.5, 'size': 3}, 'np.random.multinomial': {'n': 1, 'pvals': (1,), 'size': 3}, 'np.random.negative_binomial': {'n': 1, 'p': 0.5}, 'np.random.noncentral_chisquare': {'df': 1.0, 'nonc': 1.0, 'size': 3}, 'np.random.normal': {'loc': 0.0, 'scale': 1.0, 'size': 3}, 'np.random.pareto': {'a': 2.0, 'size': 3}, 'np.random.poisson': {'lam': 1.0, 'size': 3}, 'np.random.power': {'a': 2.0, 'size': 3}, 'np.random.randint': {'low': 1, 'high': 2, 'size': 3}, 'np.random.random': {'size': 3}, 'np.random.random_sample': {'size': 3}, 'np.random.ranf': {'size': 3}, 'np.random.rayleigh': {'scale': 1.0, 'size': 3}, 'np.random.sample': {'size': 3}, 'np.random.seed': {'seed': 4}, 'np.random.standard_cauchy': {'size': 3}, 'np.random.standard_exponential': {'size': 3}, 'np.random.standard_gamma': {'shape': 2.0, 'size': 3}, 'np.random.standard_normal': {'size': 3}, 'np.random.standard_t': {'df': 2.0, 'size': 3}, 'np.random.triangular': {'left': 1.0, 'mode': 2.0, 'right': 3.0, 'size': 3}, 'np.random.uniform': {'low': 1.0, 'high': 2.0, 'size': 3}, 'np.random.vonmises': {'mu': 1.0, 'kappa': 2.0, 'size': 3}, 'np.random.wald': {'mean': 1.0, 'scale': 2.0, 'size': 3}, 'np.random.weibull': {'a': 1.0, 'size': 3}, 'np.random.zipf': {'a': 2.0, 'size': 3}}
    for fn, args in data.items():
        argstr = ', '.join([f'{k}={v}' for k, v in args.items()])
        template = dedent(f'\n                def foo():\n                    return {fn}({argstr})\n                ')
        l = {}
        exec(template, {'np': np}, l)
        func = l['foo']
        func()
        njit(func).compile(())