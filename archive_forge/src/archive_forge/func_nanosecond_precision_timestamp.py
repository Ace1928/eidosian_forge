from __future__ import annotations
from enum import Enum
from typing import Literal
import pandas as pd
from packaging.version import Version
from xarray.coding import cftime_offsets
def nanosecond_precision_timestamp(*args, **kwargs) -> pd.Timestamp:
    """Return a nanosecond-precision Timestamp object.

    Note this function should no longer be needed after addressing GitHub issue
    #7493.
    """
    if Version(pd.__version__) >= Version('2.0.0'):
        return pd.Timestamp(*args, **kwargs).as_unit('ns')
    else:
        return pd.Timestamp(*args, **kwargs)