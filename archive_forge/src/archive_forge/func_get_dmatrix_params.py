import logging
from typing import Dict, Optional
import xgboost as xgb
import modin.pandas as pd
from modin.config import Engine
from modin.distributed.dataframe.pandas import unwrap_partitions
def get_dmatrix_params(self):
    """
        Get dict of DMatrix parameters excluding `self.data`/`self.label`.

        Returns
        -------
        dict
        """
    dmatrix_params = {'feature_names': self.feature_names, 'feature_types': self.feature_types, 'missing': self.missing, 'silent': self.silent, 'feature_weights': self.feature_weights, 'enable_categorical': self.enable_categorical}
    return dmatrix_params