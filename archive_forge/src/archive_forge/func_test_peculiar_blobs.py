import os
import numpy as np
import pytest
from ... import from_emcee
from ..helpers import _emcee_lnprior as emcee_lnprior
from ..helpers import _emcee_lnprob as emcee_lnprob
from ..helpers import (  # pylint: disable=unused-import
def test_peculiar_blobs(self, data):
    sampler = emcee.EnsembleSampler(6, 1, lambda x: (-x ** 2, (np.random.normal(x), 3)))
    sampler.run_mcmc(np.random.normal(size=(6, 1)), 20)
    inference_data = from_emcee(sampler, blob_names=['normal', 'threes'])
    fails = check_multiple_attrs({'log_likelihood': ['normal', 'threes']}, inference_data)
    assert not fails
    inference_data = from_emcee(data.obj, blob_names=['mix'])
    fails = check_multiple_attrs({'log_likelihood': ['mix']}, inference_data)
    assert not fails