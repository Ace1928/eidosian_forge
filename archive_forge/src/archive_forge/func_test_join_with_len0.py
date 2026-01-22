import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_join_with_len0(self, target_source):
    target, source = target_source
    merged = target.join(source.reindex([]), on='C')
    for col in source:
        assert col in merged
        assert merged[col].isna().all()
    merged2 = target.join(source.reindex([]), on='C', how='inner')
    tm.assert_index_equal(merged2.columns, merged.columns)
    assert len(merged2) == 0