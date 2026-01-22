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
def to_ray_dataset(modin_obj):
    """
    Convert a Modin DataFrame/Series to a Ray Dataset.

    Deprecated.

    Parameters
    ----------
    modin_obj : modin.pandas.DataFrame, modin.pandas.Series
        The DataFrame/Series to convert.

    Returns
    -------
    ray.data.Dataset
        Converted object with type depending on input.

    Notes
    -----
    Modin DataFrame/Series can only be converted to a Ray Dataset if Modin uses a Ray engine.
    """
    warnings.warn('`modin.pandas.io.to_ray_dataset` is deprecated and will be removed in a future version. ' + 'Please use `modin.pandas.io.to_ray` instead.', category=FutureWarning)
    to_ray(modin_obj)