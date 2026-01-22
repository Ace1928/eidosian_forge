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
def set_train_data(self, *X_y, add_as_eval_method=None, **dmatrix_kwargs):
    """
        Set train data for actor.

        Parameters
        ----------
        *X_y : iterable
            Sequence of ray.ObjectRef objects. First half of sequence is for
            `X` data, second for `y`. When it is passed in actor, auto-materialization
            of ray.ObjectRef -> pandas.DataFrame happens.
        add_as_eval_method : str, optional
            Name of eval data. Used in case when train data also used for evaluation.
        **dmatrix_kwargs : dict
            Keyword parameters for ``xgb.DMatrix``.
        """
    self._dtrain = self._get_dmatrix(X_y, **dmatrix_kwargs)
    if add_as_eval_method is not None:
        self._evals.append((self._dtrain, add_as_eval_method))