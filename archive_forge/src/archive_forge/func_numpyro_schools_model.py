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
def numpyro_schools_model(data, draws, chains):
    """Centered eight schools implementation in NumPyro."""
    from jax.random import PRNGKey
    from numpyro.infer import MCMC, NUTS
    mcmc = MCMC(NUTS(_numpyro_noncentered_model), num_warmup=draws, num_samples=draws, num_chains=chains, chain_method='sequential')
    mcmc.run(PRNGKey(0), extra_fields=('num_steps', 'energy'), **data)
    mcmc.sampler._sample_fn = None
    mcmc.sampler._init_fn = None
    mcmc.sampler._postprocess_fn = None
    mcmc.sampler._potential_fn = None
    mcmc.sampler._potential_fn_gen = None
    mcmc._cache = {}
    return mcmc