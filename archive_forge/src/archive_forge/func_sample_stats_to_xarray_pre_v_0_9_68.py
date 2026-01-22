import logging
import re
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
import numpy as np
from ..rcparams import rcParams
from .base import dict_to_dataset, infer_stan_dtypes, make_attrs, requires
from .inference_data import InferenceData
def sample_stats_to_xarray_pre_v_0_9_68(self, fit):
    """Extract sample_stats from fit."""
    dtypes = {'divergent__': bool, 'n_leapfrog__': np.int64, 'treedepth__': np.int64}
    columns = fit.column_names
    valid_cols = [col for col in columns if col.endswith('__')]
    data, data_warmup = _unpack_frame(fit, columns, valid_cols, self.save_warmup, self.dtypes)
    for s_param in list(data.keys()):
        s_param_, *_ = s_param.split('.')
        name = re.sub('__$', '', s_param_)
        name = 'diverging' if name == 'divergent' else name
        data[name] = data.pop(s_param).astype(dtypes.get(s_param, float))
        if data_warmup:
            data_warmup[name] = data_warmup.pop(s_param).astype(dtypes.get(s_param, float))
    return (dict_to_dataset(data, library=self.cmdstanpy, coords=self.coords, dims=self.dims, index_origin=self.index_origin), dict_to_dataset(data_warmup, library=self.cmdstanpy, coords=self.coords, dims=self.dims, index_origin=self.index_origin))