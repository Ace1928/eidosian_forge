import abc
import warnings
from typing import Hashable, List, Optional
import numpy as np
import pandas
import pandas.core.resample
from pandas._typing import DtypeBackend, IndexLabel, Suffixes
from pandas.core.dtypes.common import is_number, is_scalar
from modin.config import StorageFormat
from modin.core.dataframe.algebra.default2pandas import (
from modin.error_message import ErrorMessage
from modin.logging import ClassLogger
from modin.utils import MODIN_UNNAMED_SERIES_LABEL, try_cast_to_pandas
from . import doc_utils
@doc_utils.add_refer_to('Series.dt.asfreq')
def dt_asfreq(self, freq=None, how: str='E'):
    """
        Convert the PeriodArray to the specified frequency `freq`.

        Equivalent to applying pandas.Period.asfreq() with the given arguments to each Period in this PeriodArray.

        Parameters
        ----------
        freq : str, optional
            A frequency.
        how : str {'E', 'S'}, default: 'E'
            Whether the elements should be aligned to the end or start within pa period.
            * 'E', "END", or "FINISH" for end,
            * 'S', "START", or "BEGIN" for start.
            January 31st ("END") vs. January 1st ("START") for example.

        Returns
        -------
        BaseQueryCompiler
            New QueryCompiler containing period data.
        """
    return DateTimeDefault.register(pandas.Series.dt.asfreq)(self, freq, how)