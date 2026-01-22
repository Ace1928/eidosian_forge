from __future__ import annotations
from textwrap import dedent
import numpy as np
import pandas as pd
import pytest
import dask.array as da
import dask.dataframe as dd
from dask.dataframe.utils import get_string_dtype, pyarrow_strings_enabled
from dask.utils import maybe_pluralize
def test_repr_meta_mutation():
    df = pd.DataFrame({'a': range(5), 'b': ['a', 'b', 'c', 'd', 'e']})
    ddf = dd.from_pandas(df, npartitions=2)
    s1 = repr(ddf)
    assert repr(ddf) == s1
    ddf.b = ddf.b.astype('category')
    assert repr(ddf) != s1