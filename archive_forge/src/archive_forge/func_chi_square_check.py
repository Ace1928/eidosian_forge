import time
import gzip
import struct
import traceback
import numbers
import sys
import os
import platform
import errno
import logging
import bz2
import zipfile
import json
from contextlib import contextmanager
from collections import OrderedDict
import numpy as np
import numpy.testing as npt
import numpy.random as rnd
import mxnet as mx
from .context import Context, current_context
from .ndarray.ndarray import _STORAGE_TYPE_STR_TO_ID
from .ndarray import array
from .symbol import Symbol
from .symbol.numpy import _Symbol as np_symbol
from .util import use_np, getenv, setenv  # pylint: disable=unused-import
from .runtime import Features
from .numpy_extension import get_cuda_compute_capability
def chi_square_check(generator, buckets, probs, nsamples=1000000):
    """Run the chi-square test for the generator. The generator can be both continuous and discrete.

    If the generator is continuous, the buckets should contain tuples of (range_min, range_max)     and the probs should be the corresponding ideal probability within the specific ranges.     Otherwise, the buckets should contain all the possible values generated over the discrete distribution and the     probs should be groud-truth probability.

    Usually the user is required to specify the probs parameter.

    After obtaining the p value, we could further use the standard p > 0.05 (alpha) threshold to get     the final result.

    Examples::

      buckets, probs = gen_buckets_probs_with_ppf(lambda x: ss.norm.ppf(x, 0, 1), 5)
      generator = lambda x: np.random.normal(0, 1.0, size=x)
      p = chi_square_check(generator=generator, buckets=buckets, probs=probs)
      assert(p > 0.05)

    Parameters
    ----------
    generator: function
        A function that is assumed to generate i.i.d samples from a specific distribution.
        generator(N) should generate N random samples.
    buckets: list of tuple or list of number
        The buckets to run the chi-square the test. Make sure that the buckets cover
        the whole range of the distribution. Also, the buckets must be in ascending order and have
        no intersection
    probs: list or tuple
        The ground-truth probability of the random value fall in a specific bucket.
    nsamples:int
        The number of samples to generate for the testing

    Returns
    -------
    p : float
        p value that the generator has the expected distribution.
        A higher value indicates a larger confidence
    obs_freq : list
        Observed frequency of buckets
    expected_freq : list
        The expected (ground-truth) frequency of the buckets
    """
    if not ss:
        raise ImportError('scipy is not available. Please check if the scipy python bindings are installed.')
    assert isinstance(buckets, list)
    samples = generator(nsamples)
    assert len(probs) == len(buckets)
    if isinstance(buckets[0], (list, tuple)):
        continuous_dist = True
        buckets_npy = np.zeros((len(buckets) * 2,), dtype=np.float32)
        for i, _ in enumerate(buckets):
            assert buckets[i][0] <= buckets[i][1]
            if i < len(buckets) - 1:
                assert buckets[i][1] <= buckets[i + 1][0]
            buckets_npy[i * 2] = buckets[i][0]
            buckets_npy[i * 2 + 1] = buckets[i][1]
    else:
        continuous_dist = False
    expected_freq = (nsamples * np.array(probs, dtype=np.float32)).astype(np.int32)
    if continuous_dist:
        sample_bucket_ids = np.searchsorted(buckets_npy, samples, side='right')
    else:
        sample_bucket_ids = np.array(samples)
    if continuous_dist:
        sample_bucket_ids = sample_bucket_ids // 2
    obs_freq = np.zeros(shape=len(buckets), dtype=np.int)
    for i, _ in enumerate(buckets):
        if continuous_dist:
            obs_freq[i] = (sample_bucket_ids == i).sum()
        else:
            obs_freq[i] = (sample_bucket_ids == buckets[i]).sum()
    _, p = ss.chisquare(f_obs=obs_freq, f_exp=expected_freq)
    return (p, obs_freq, expected_freq)