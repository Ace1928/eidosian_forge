import abc
from typing import Callable, Dict, Optional, Type, Union
import numpy as np
import pandas as pd
from ray.air.data_batch_type import DataBatchType
from ray.air.util.data_batch_conversion import (
from ray.data import Preprocessor
from ray.train import Checkpoint
from ray.util.annotations import DeveloperAPI, PublicAPI
@classmethod
def from_pandas_udf(cls, pandas_udf: Callable[[pd.DataFrame], pd.DataFrame]) -> 'Predictor':
    """Create a Predictor from a Pandas UDF.

        Args:
            pandas_udf: A function that takes a pandas.DataFrame and other
                optional kwargs and returns a pandas.DataFrame.
        """

    class PandasUDFPredictor(Predictor):

        @classmethod
        def from_checkpoint(cls, checkpoint: Checkpoint, **kwargs) -> 'Predictor':
            return PandasUDFPredictor()

        def _predict_pandas(self, df, **kwargs) -> 'pd.DataFrame':
            return pandas_udf(df, **kwargs)
    return PandasUDFPredictor()