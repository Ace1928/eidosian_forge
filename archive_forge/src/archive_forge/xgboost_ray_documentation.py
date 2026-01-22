import logging
import math
import time
import warnings
from collections import defaultdict
from typing import Dict, List
import numpy as np
import pandas
import ray
import xgboost as xgb
from ray.util import get_node_ip_address
from modin.core.execution.ray.common import RayWrapper
from modin.distributed.dataframe.pandas import from_partitions
from .utils import RabitContext, RabitContextManager

        Run local XGBoost training.

        Connects to Rabit Tracker environment to share training data between
        actors and trains XGBoost booster using `self._dtrain`.

        Parameters
        ----------
        rabit_args : list
            List with environment variables for Rabit Tracker.
        params : dict
            Booster params.
        *args : iterable
            Other parameters for `xgboost.train`.
        **kwargs : dict
            Other parameters for `xgboost.train`.

        Returns
        -------
        dict
            A dictionary with trained booster and dict of
            evaluation results
            as {"booster": xgb.Booster, "history": dict}.
        