from __future__ import annotations
import os
import pathlib
from typing import TYPE_CHECKING
import numpy as np
from xarray.backends.api import open_dataset as _open_dataset
from xarray.core.dataarray import DataArray
from xarray.core.dataset import Dataset

    Create an example dataset.

    Parameters
    ----------
    seed : int, optional
        Seed for the random number generation.
    