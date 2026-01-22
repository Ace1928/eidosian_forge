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
def check_inf(rng: RNG) -> None:
    """Validate there's no inf in X."""
    X = rng.random(size=32).reshape(8, 4)
    y = rng.random(size=8)
    X[5, 2] = np.inf
    with pytest.raises(ValueError, match='Input data contains `inf`'):
        xgboost.QuantileDMatrix(X, y)
    with pytest.raises(ValueError, match='Input data contains `inf`'):
        xgboost.DMatrix(X, y)