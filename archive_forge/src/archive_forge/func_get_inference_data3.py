import os
import sys
import tempfile
from glob import glob
import numpy as np
import pytest
from ... import from_cmdstanpy
from ..helpers import (  # pylint: disable=unused-import
def get_inference_data3(self, data, eight_schools_params):
    """multiple vars as lists."""
    return from_cmdstanpy(posterior=data.obj, posterior_predictive=['y_hat', 'log_lik'], prior=data.obj, prior_predictive=['y_hat', 'log_lik'], observed_data={'y': eight_schools_params['y']}, coords={'school': np.arange(eight_schools_params['J']), 'half school': ['a', 'b', 'c', 'd'], 'extra_dim': ['x', 'y']}, dims={'eta': ['extra_dim', 'half school'], 'y': ['school'], 'y_hat': ['school'], 'theta': ['school'], 'log_lik': ['log_lik_dim']}, dtypes=data.model)