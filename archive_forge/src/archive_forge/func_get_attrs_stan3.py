import re
from collections import OrderedDict
from copy import deepcopy
from math import ceil
import numpy as np
import xarray as xr
from .. import _log
from ..rcparams import rcParams
from .base import dict_to_dataset, generate_dims_coords, infer_stan_dtypes, make_attrs, requires
from .inference_data import InferenceData
def get_attrs_stan3(fit, model=None):
    """Get attributes from PyStan3 fit and model object."""
    attrs = {}
    for key in ['num_chains', 'num_samples', 'num_thin', 'num_warmup', 'save_warmup']:
        try:
            attrs[key] = getattr(fit, key)
        except AttributeError as exp:
            _log.warning('Failed to access attribute %s in fit object %s', key, exp)
    if model is not None:
        for key in ['model_name', 'program_code', 'random_seed']:
            try:
                attrs[key] = getattr(model, key)
            except AttributeError as exp:
                _log.warning('Failed to access attribute %s in model object %s', key, exp)
    return attrs