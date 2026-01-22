import os
import numpy as np
import pytest
from ... import from_emcee
from ..helpers import _emcee_lnprior as emcee_lnprior
from ..helpers import _emcee_lnprob as emcee_lnprob
from ..helpers import (  # pylint: disable=unused-import
def test_no_blobs_error(self):
    sampler = emcee.EnsembleSampler(6, 1, lambda x: -x ** 2)
    sampler.run_mcmc(np.random.normal(size=(6, 1)), 20)
    with pytest.raises(ValueError):
        from_emcee(sampler, blob_names=['inexistent'])