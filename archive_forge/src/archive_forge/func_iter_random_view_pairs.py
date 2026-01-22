import itertools
import pytest
import numpy as np
from numpy.core._multiarray_tests import solve_diophantine, internal_overlap
from numpy.core import _umath_tests
from numpy.lib.stride_tricks import as_strided
from numpy.testing import (
def iter_random_view_pairs(x, same_steps=True, equal_size=False):
    rng = np.random.RandomState(1234)
    if equal_size and same_steps:
        raise ValueError()

    def random_slice(n, step):
        start = rng.randint(0, n + 1, dtype=np.intp)
        stop = rng.randint(start, n + 1, dtype=np.intp)
        if rng.randint(0, 2, dtype=np.intp) == 0:
            stop, start = (start, stop)
            step *= -1
        return slice(start, stop, step)

    def random_slice_fixed_size(n, step, size):
        start = rng.randint(0, n + 1 - size * step)
        stop = start + (size - 1) * step + 1
        if rng.randint(0, 2) == 0:
            stop, start = (start - 1, stop - 1)
            if stop < 0:
                stop = None
            step *= -1
        return slice(start, stop, step)
    yield (x, x)
    for j in range(1, 7, 3):
        yield (x[j:], x[:-j])
        yield (x[..., j:], x[..., :-j])
    strides = list(x.strides)
    strides[0] = 0
    xp = as_strided(x, shape=x.shape, strides=strides)
    yield (x, xp)
    yield (xp, xp)
    strides = list(x.strides)
    if strides[0] > 1:
        strides[0] = 1
    xp = as_strided(x, shape=x.shape, strides=strides)
    yield (x, xp)
    yield (xp, xp)
    while True:
        steps = tuple((rng.randint(1, 11, dtype=np.intp) if rng.randint(0, 5, dtype=np.intp) == 0 else 1 for j in range(x.ndim)))
        s1 = tuple((random_slice(p, s) for p, s in zip(x.shape, steps)))
        t1 = np.arange(x.ndim)
        rng.shuffle(t1)
        if equal_size:
            t2 = t1
        else:
            t2 = np.arange(x.ndim)
            rng.shuffle(t2)
        a = x[s1]
        if equal_size:
            if a.size == 0:
                continue
            steps2 = tuple((rng.randint(1, max(2, p // (1 + pa))) if rng.randint(0, 5) == 0 else 1 for p, s, pa in zip(x.shape, s1, a.shape)))
            s2 = tuple((random_slice_fixed_size(p, s, pa) for p, s, pa in zip(x.shape, steps2, a.shape)))
        elif same_steps:
            steps2 = steps
        else:
            steps2 = tuple((rng.randint(1, 11, dtype=np.intp) if rng.randint(0, 5, dtype=np.intp) == 0 else 1 for j in range(x.ndim)))
        if not equal_size:
            s2 = tuple((random_slice(p, s) for p, s in zip(x.shape, steps2)))
        a = a.transpose(t1)
        b = x[s2].transpose(t2)
        yield (a, b)