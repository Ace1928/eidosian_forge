import base64
import sys
from typing import Iterable, Tuple, Union
import numpy as np
import pandas as pd
from packaging.version import Version
from mlflow.exceptions import MlflowException
from mlflow.protos import facet_feature_statistics_pb2
from mlflow.recipes.cards import histogram_generator
def datetime_and_timedelta_converter(dtype):
    """
    Converts a Numpy dtype to a converter method if applicable.
    The converter method takes in a numpy array of objects of the provided
    dtype and returns a numpy array of the numbers backing that object for
    statistical analysis. Returns None if no converter is necessary.

    Args:
        dtype: The numpy dtype to make a converter for.

    Returns:
        The converter method or None.

    """
    if np.issubdtype(dtype, np.datetime64):

        def datetime_converter(dt_list):
            return np.array([pd.Timestamp(dt).value for dt in dt_list])
        return datetime_converter
    elif np.issubdtype(dtype, np.timedelta64):

        def timedelta_converter(td_list):
            return np.array([pd.Timedelta(td).value for td in td_list])
        return timedelta_converter
    else:
        return None