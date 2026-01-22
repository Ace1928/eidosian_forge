import typing as tp
from collections import OrderedDict
from collections.abc import Iterable
import numpy as np
import xarray
from .inference_data import InferenceData
from ..rcparams import rcParams
from .base import dict_to_dataset
def to_inference_data(self):
    """Convert all available data to an InferenceData object."""
    save_warmup = self.save_warmup and self.warmup_iterations > 0
    idata_dict = {'posterior': self.posterior_to_xarray(), 'prior': self.prior_to_xarray(), 'log_likelihood': self.log_likelihood_to_xarray(), 'save_warmup': save_warmup}
    return InferenceData(**idata_dict)