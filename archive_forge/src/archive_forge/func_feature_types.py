import logging
from typing import Dict, Optional
import xgboost as xgb
import modin.pandas as pd
from modin.config import Engine
from modin.distributed.dataframe.pandas import unwrap_partitions
@feature_types.setter
def feature_types(self, feature_types):
    """
        Set column types.

        Parameters
        ----------
        feature_types : list or None
            Labels for columns. In case None, existing feature names will be reset.
        """
    if feature_types is not None:
        if not isinstance(feature_types, (list, str)):
            raise TypeError('feature_types must be string or list of strings')
        if isinstance(feature_types, str):
            feature_types = [feature_types] * self.num_col()
            feature_types = list(feature_types) if not isinstance(feature_types, str) else [feature_types]
    else:
        feature_types = None
    self._feature_types = feature_types