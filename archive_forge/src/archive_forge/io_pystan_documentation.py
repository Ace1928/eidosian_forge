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
Convert all available data to an InferenceData object.

        Note that if groups can not be created (i.e., there is no `fit`, so
        the `posterior` and `sample_stats` can not be extracted), then the InferenceData
        will not have those groups.
        