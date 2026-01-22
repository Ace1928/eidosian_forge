import logging
from typing import Dict, Optional
import xgboost as xgb
import modin.pandas as pd
from modin.config import Engine
from modin.distributed.dataframe.pandas import unwrap_partitions
def get_float_info(self, name):
    """
        Get float property from the DMatrix.

        Parameters
        ----------
        name : str
            The field name of the information.

        Returns
        -------
        A NumPy array of float information of the data.
        """
    return getattr(self, name)