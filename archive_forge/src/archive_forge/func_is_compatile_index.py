from datetime import datetime
from typing import (
import numpy as np
import pandas as pd
import pyarrow as pa
from pandas.core.frame import DataFrame
from triad.utils.assertion import assert_or_throw
from triad.utils.pyarrow import (
def is_compatile_index(self, df: T) -> bool:
    """Check whether the datafame is compatible with the operations inside
        this utils collection

        :param df: pandas like dataframe
        :return: if it is compatible
        """
    return isinstance(df.index, pd.RangeIndex) or df.index.inferred_type == 'integer'