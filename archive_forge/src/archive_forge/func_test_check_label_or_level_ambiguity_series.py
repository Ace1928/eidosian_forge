import pytest
from pandas.core.dtypes.missing import array_equivalent
import pandas as pd
def test_check_label_or_level_ambiguity_series(df):
    s = df.set_index('L1').L2
    s._check_label_or_level_ambiguity('L1', axis=0)
    s._check_label_or_level_ambiguity('L2', axis=0)
    s = df.set_index(['L1', 'L2']).L3
    s._check_label_or_level_ambiguity('L1', axis=0)
    s._check_label_or_level_ambiguity('L2', axis=0)
    s._check_label_or_level_ambiguity('L3', axis=0)