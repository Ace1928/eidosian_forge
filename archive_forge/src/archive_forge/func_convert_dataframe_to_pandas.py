from __future__ import annotations
from collections.abc import Mapping, Sized
from typing import cast
import warnings
import pandas as pd
from pandas import DataFrame
from seaborn._core.typing import DataSource, VariableSpec, ColumnName
from seaborn.utils import _version_predates
def convert_dataframe_to_pandas(data: object) -> pd.DataFrame:
    """Use the DataFrame exchange protocol, or fail gracefully."""
    if isinstance(data, pd.DataFrame):
        return data
    if not hasattr(pd.api, 'interchange'):
        msg = 'Support for non-pandas DataFrame objects requires a version of pandas that implements the DataFrame interchange protocol. Please upgrade your pandas version or coerce your data to pandas before passing it to seaborn.'
        raise TypeError(msg)
    if _version_predates(pd, '2.0.2'):
        msg = f'DataFrame interchange with pandas<2.0.2 has some known issues. You are using pandas {pd.__version__}. Continuing, but it is recommended to carefully inspect the results and to consider upgrading.'
        warnings.warn(msg, stacklevel=2)
    try:
        return pd.api.interchange.from_dataframe(data)
    except Exception as err:
        msg = 'Encountered an exception when converting data source to a pandas DataFrame. See traceback above for details.'
        raise RuntimeError(msg) from err