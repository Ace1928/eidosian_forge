import os
from collections import Counter
from itertools import combinations, product
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_array_equal
from scipy.spatial import distance
from scipy.stats import shapiro
from scipy.stats._sobol import _test_find_index
from scipy.stats import qmc
from scipy.stats._qmc import (
def test_discrepancy_alternative_implementation(self):
    """Alternative definitions from Matt Haberland."""

    def disc_c2(x):
        n, s = x.shape
        xij = x
        disc1 = np.sum(np.prod(1 + 1 / 2 * np.abs(xij - 0.5) - 1 / 2 * np.abs(xij - 0.5) ** 2, axis=1))
        xij = x[None, :, :]
        xkj = x[:, None, :]
        disc2 = np.sum(np.sum(np.prod(1 + 1 / 2 * np.abs(xij - 0.5) + 1 / 2 * np.abs(xkj - 0.5) - 1 / 2 * np.abs(xij - xkj), axis=2), axis=0))
        return (13 / 12) ** s - 2 / n * disc1 + 1 / n ** 2 * disc2

    def disc_wd(x):
        n, s = x.shape
        xij = x[None, :, :]
        xkj = x[:, None, :]
        disc = np.sum(np.sum(np.prod(3 / 2 - np.abs(xij - xkj) + np.abs(xij - xkj) ** 2, axis=2), axis=0))
        return -(4 / 3) ** s + 1 / n ** 2 * disc

    def disc_md(x):
        n, s = x.shape
        xij = x
        disc1 = np.sum(np.prod(5 / 3 - 1 / 4 * np.abs(xij - 0.5) - 1 / 4 * np.abs(xij - 0.5) ** 2, axis=1))
        xij = x[None, :, :]
        xkj = x[:, None, :]
        disc2 = np.sum(np.sum(np.prod(15 / 8 - 1 / 4 * np.abs(xij - 0.5) - 1 / 4 * np.abs(xkj - 0.5) - 3 / 4 * np.abs(xij - xkj) + 1 / 2 * np.abs(xij - xkj) ** 2, axis=2), axis=0))
        return (19 / 12) ** s - 2 / n * disc1 + 1 / n ** 2 * disc2

    def disc_star_l2(x):
        n, s = x.shape
        return np.sqrt(3 ** (-s) - 2 ** (1 - s) / n * np.sum(np.prod(1 - x ** 2, axis=1)) + np.sum([np.prod(1 - np.maximum(x[k, :], x[j, :])) for k in range(n) for j in range(n)]) / n ** 2)
    rng = np.random.default_rng(117065081482921065782761407107747179201)
    sample = rng.random((30, 10))
    disc_curr = qmc.discrepancy(sample, method='CD')
    disc_alt = disc_c2(sample)
    assert_allclose(disc_curr, disc_alt)
    disc_curr = qmc.discrepancy(sample, method='WD')
    disc_alt = disc_wd(sample)
    assert_allclose(disc_curr, disc_alt)
    disc_curr = qmc.discrepancy(sample, method='MD')
    disc_alt = disc_md(sample)
    assert_allclose(disc_curr, disc_alt)
    disc_curr = qmc.discrepancy(sample, method='L2-star')
    disc_alt = disc_star_l2(sample)
    assert_allclose(disc_curr, disc_alt)