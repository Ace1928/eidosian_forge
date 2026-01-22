from typing import Callable, Dict, Optional, Tuple, Union, TYPE_CHECKING
from copy import deepcopy
import logging
import numpy as np
import pandas as pd
from ray.tune import TuneError
from ray.tune.experiment import Trial
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.schedulers.pbt import _PBTTrialState
from ray.tune.utils.util import flatten_dict, unflatten_dict
from ray.util.debug import log_once
def import_pb2_dependencies():
    try:
        import GPy
    except ImportError:
        GPy = None
    try:
        import sklearn
    except ImportError:
        sklearn = None
    return (GPy, sklearn)