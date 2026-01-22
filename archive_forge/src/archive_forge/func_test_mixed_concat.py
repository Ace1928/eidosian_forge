import numpy as np
import pandas
import pytest
import modin.pandas as pd
from modin.config import NPartitions, StorageFormat
from modin.pandas.utils import from_pandas
from modin.utils import get_current_execution
from .utils import (
def test_mixed_concat():
    df, df2 = generate_dfs()
    df3 = df.copy()
    mixed_dfs = [from_pandas(df), from_pandas(df2), df3]
    df_equals(pd.concat(mixed_dfs), pandas.concat([df, df2, df3]))