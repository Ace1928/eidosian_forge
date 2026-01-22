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
def pyro_noncentered_schools(data, draws, chains):
    """Non-centered eight schools implementation in Pyro."""
    import torch
    from pyro.infer import MCMC, NUTS
    y = torch.from_numpy(data['y']).float()
    sigma = torch.from_numpy(data['sigma']).float()
    nuts_kernel = NUTS(_pyro_noncentered_model, jit_compile=True, ignore_jit_warnings=True)
    posterior = MCMC(nuts_kernel, num_samples=draws, warmup_steps=draws, num_chains=chains)
    posterior.run(data['J'], sigma, y)
    posterior.sampler = None
    posterior.kernel.potential_fn = None
    return posterior