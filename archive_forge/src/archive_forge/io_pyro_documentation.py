import logging
from typing import Callable, Optional
import warnings
import numpy as np
from packaging import version
from .. import utils
from ..rcparams import rcParams
from .base import dict_to_dataset, requires
from .inference_data import InferenceData
Convert all available data to an InferenceData object.