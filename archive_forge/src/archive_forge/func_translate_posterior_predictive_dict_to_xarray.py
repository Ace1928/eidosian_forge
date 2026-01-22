import logging
from typing import Callable, Optional
import warnings
import numpy as np
from packaging import version
from .. import utils
from ..rcparams import rcParams
from .base import dict_to_dataset, requires
from .inference_data import InferenceData
def translate_posterior_predictive_dict_to_xarray(self, dct, dims):
    """Convert posterior_predictive or prediction samples to xarray."""
    data = {}
    for k, ary in dct.items():
        ary = ary.detach().cpu().numpy()
        shape = ary.shape
        if shape[0] == self.nchains and shape[1] == self.ndraws:
            data[k] = ary
        elif shape[0] == self.nchains * self.ndraws:
            data[k] = ary.reshape((self.nchains, self.ndraws, *shape[1:]))
        else:
            data[k] = utils.expand_dims(ary)
            _log.warning('posterior predictive shape not compatible with number of chains and draws.This can mean that some draws or even whole chains are not represented.')
    return dict_to_dataset(data, library=self.pyro, coords=self.coords, dims=dims)