from __future__ import annotations
import re
from math import ceil
from typing import Generator, Hashable, List, Optional
import numpy as np
import pandas
from modin.config import MinPartitionSize, NPartitions
def width_fn_pandas(df):
    """
    Compute number of columns of passed `pandas.DataFrame`.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    int
    """
    assert isinstance(df, pandas.DataFrame)
    return len(df.columns) if len(df.columns) > 0 else 0