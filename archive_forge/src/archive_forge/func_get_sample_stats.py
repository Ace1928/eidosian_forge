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
def get_sample_stats(fit, warmup=False, dtypes=None):
    """Extract sample stats from PyStan fit."""
    if dtypes is None:
        dtypes = {}
    dtypes = {'divergent__': bool, 'n_leapfrog__': np.int64, 'treedepth__': np.int64, **dtypes}
    rename_dict = {'divergent': 'diverging', 'n_leapfrog': 'n_steps', 'treedepth': 'tree_depth', 'stepsize': 'step_size', 'accept_stat': 'acceptance_rate'}
    ndraws_warmup = fit.sim['warmup2']
    if max(ndraws_warmup) == 0:
        warmup = False
    ndraws = [s - w for s, w in zip(fit.sim['n_save'], ndraws_warmup)]
    extraction = OrderedDict()
    extraction_warmup = OrderedDict()
    for chain, (pyholder, ndraw, ndraw_warmup) in enumerate(zip(fit.sim['samples'], ndraws, ndraws_warmup)):
        if chain == 0:
            for key in pyholder['sampler_param_names']:
                extraction[key] = []
                if warmup:
                    extraction_warmup[key] = []
        for key, values in zip(pyholder['sampler_param_names'], pyholder['sampler_params']):
            extraction[key].append(values[-ndraw:])
            if warmup:
                extraction_warmup[key].append(values[:ndraw_warmup])
    data = OrderedDict()
    for key, values in extraction.items():
        values = np.stack(values, axis=0)
        dtype = dtypes.get(key)
        values = values.astype(dtype)
        name = re.sub('__$', '', key)
        name = rename_dict.get(name, name)
        data[name] = values
    data_warmup = OrderedDict()
    if warmup:
        for key, values in extraction_warmup.items():
            values = np.stack(values, axis=0)
            values = values.astype(dtypes.get(key))
            name = re.sub('__$', '', key)
            name = rename_dict.get(name, name)
            data_warmup[name] = values
    return (data, data_warmup)