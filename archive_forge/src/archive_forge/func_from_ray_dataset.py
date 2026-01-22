from __future__ import annotations
import csv
import inspect
import pathlib
import pickle
import warnings
from typing import (
import numpy as np
import pandas
from pandas._libs.lib import NoDefault, no_default
from pandas._typing import (
from pandas.io.parsers import TextFileReader
from pandas.io.parsers.readers import _c_parser_defaults
from modin.config import ExperimentalNumPyAPI
from modin.error_message import ErrorMessage
from modin.logging import ClassLogger, enable_logging
from modin.utils import (
def from_ray_dataset(ray_obj) -> DataFrame:
    """
    Convert a Ray Dataset into Modin DataFrame.

    Deprecated.

    Parameters
    ----------
    ray_obj : ray.data.Dataset
        The Ray Dataset to convert from.

    Returns
    -------
    DataFrame
        A new Modin DataFrame object.

    Notes
    -----
    Ray Dataset can only be converted to Modin DataFrame if Modin uses a Ray engine.
    """
    warnings.warn('`modin.pandas.io.from_ray_dataset` is deprecated and will be removed in a future version. ' + 'Please use `modin.pandas.io.from_ray` instead.', category=FutureWarning)
    from_ray(ray_obj)