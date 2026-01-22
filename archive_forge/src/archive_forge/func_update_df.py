from typing import Callable, Optional
import numpy as np
import modin.pandas as pd
from modin.config import NPartitions
from modin.core.execution.ray.implementations.pandas_on_ray.dataframe.dataframe import (
from modin.core.storage_formats.pandas import PandasQueryCompiler
from modin.error_message import ErrorMessage
from modin.utils import get_current_execution
def update_df(self, df):
    """
        Update the dataframe to perform this pipeline on.

        Parameters
        ----------
        df : modin.pandas.DataFrame
            The new dataframe to perform this pipeline on.
        """
    if get_current_execution() != 'PandasOnRay' or not isinstance(df._query_compiler._modin_frame, PandasOnRayDataframe):
        ErrorMessage.not_implemented('Batch Pipeline API is only implemented for `PandasOnRay` execution.')
    self.df = df