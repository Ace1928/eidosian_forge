from __future__ import annotations
from itertools import product
import pandas as pd
import pytest
import dask.dataframe as dd
from dask.dataframe._compat import PANDAS_GE_220
from dask.dataframe.utils import assert_eq
def test_resample_pads_last_division_to_avoid_off_by_one():
    times = [1545362463409128000, 1545362504369352000, 1545362545326966000, 1545363118769636000, 1545363159726490000, 1545363200687178000, 1545363241648824000, 1573318190393973000, 1573318231353350000, 1573318272313774000, 1573318313275299000, 1573318354233962000, 1573318395195456000, 1573318436154609000, 1580687544437145000, 1580687585394881000, 1580687667316809000, 1580687708275414000, 1580687790195742000, 1580687831154951000, 1580687872115363000, 1580687954035133000, 1559127673402811000]
    freq = '1QE' if PANDAS_GE_220 else '1Q'
    df = pd.DataFrame({'Time': times, 'Counts': range(len(times))})
    df['Time'] = pd.to_datetime(df['Time'], utc=True)
    expected = df.set_index('Time').resample(freq).size()
    ddf = dd.from_pandas(df, npartitions=2).set_index('Time')
    actual = ddf.resample(freq).size().compute()
    assert_eq(actual, expected)