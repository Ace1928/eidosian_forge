from datetime import datetime
from typing import (
import numpy as np
import pandas as pd
import pyarrow as pa
from pandas.core.frame import DataFrame
from triad.utils.assertion import assert_or_throw
from triad.utils.pyarrow import (
def cast_df(self, df: T, schema: pa.Schema, use_extension_types: bool=True, use_arrow_dtype: bool=False, **kwargs: Any) -> T:
    """Cast pandas like dataframe to comply with ``schema``.

        :param df: pandas like dataframe
        :param schema: pyarrow schema to cast to
        :param use_extension_types: whether to use ``ExtensionDType``, default True
        :param use_arrow_dtype: whether to use ``ArrowDtype``, default False
        :param kwargs: other arguments passed to ``pa.Table.from_pandas``

        :return: converted dataframe
        """
    dtypes = to_pandas_dtype(schema, use_extension_types=use_extension_types, use_arrow_dtype=use_arrow_dtype)
    if len(df) == 0:
        return pd.DataFrame({k: pd.Series(dtype=v) for k, v in dtypes.items()})
    if dtypes == df.dtypes.to_dict():
        return df
    adf = pa.Table.from_pandas(df, preserve_index=False, safe=False, **{'nthreads': 1, **kwargs}).replace_schema_metadata()
    adf = cast_pa_table(adf, schema)
    return pa_table_to_pandas(adf, use_extension_types=use_extension_types, use_arrow_dtype=use_arrow_dtype)