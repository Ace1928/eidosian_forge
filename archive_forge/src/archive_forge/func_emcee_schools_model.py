import gzip
import importlib
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union
import cloudpickle
import numpy as np
import pytest
from _pytest.outcomes import Skipped
from packaging.version import Version
from ..data import InferenceData, from_dict
def emcee_schools_model(data, draws, chains):
    """Schools model in emcee."""
    import emcee
    chains = 10 * chains
    y = data['y']
    sigma = data['sigma']
    J = data['J']
    ndim = J + 2
    pos = np.random.normal(size=(chains, ndim))
    pos[:, 1] = np.absolute(pos[:, 1])
    if emcee_version() < 3:
        sampler = emcee.EnsembleSampler(chains, ndim, _emcee_lnprob, args=(y, sigma))
        sampler.run_mcmc(pos, draws)
    else:
        here = os.path.dirname(os.path.abspath(__file__))
        data_directory = os.path.join(here, 'saved_models')
        filepath = os.path.join(data_directory, 'reader_testfile.h5')
        backend = emcee.backends.HDFBackend(filepath)
        backend.reset(chains, ndim)
        sampler = emcee.EnsembleSampler(chains, ndim, _emcee_lnprob, args=(y, sigma), backend=backend)
        sampler.run_mcmc(pos, draws, store=True)
    return sampler