import os
import zipfile
from dataclasses import dataclass
from typing import Any, Generator, List, NamedTuple, Optional, Tuple, Union
from urllib import request
import numpy as np
import pytest
from numpy import typing as npt
from numpy.random import Generator as RNG
from scipy import sparse
import xgboost
from xgboost.data import pandas_pyarrow_mapper
@memory.cache
def get_california_housing() -> Tuple[np.ndarray, np.ndarray]:
    """Fetch the California housing dataset from sklearn."""
    datasets = pytest.importorskip('sklearn.datasets')
    data = datasets.fetch_california_housing()
    return (data.data, data.target)