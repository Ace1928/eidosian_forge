import os
import sys
import tempfile
from glob import glob
import numpy as np
import pytest
from ... import from_cmdstanpy
from ..helpers import (  # pylint: disable=unused-import
def get_inference_data2(self, data, eight_schools_params):
    """vars as lists."""
    return from_cmdstanpy(posterior=data.obj, posterior_predictive=['y_hat'], predictions=['y_hat', 'log_lik'], prior=data.obj, prior_predictive=['y_hat'], observed_data={'y': eight_schools_params['y']}, constant_data=eight_schools_params, predictions_constant_data=eight_schools_params, log_likelihood=['log_lik', 'y_hat'], coords={'school': np.arange(eight_schools_params['J']), 'log_lik_dim': np.arange(eight_schools_params['J'])}, dims={'eta': ['extra_dim', 'half school'], 'y': ['school'], 'y_hat': ['school'], 'theta': ['school'], 'log_lik': ['log_lik_dim']})