from datetime import datetime
import re
import numpy as np
import pytest
import pandas as pd
from pandas import (
from pandas.tests.strings import (
def test_rsplit_to_multiindex_expand_no_split():
    idx = Index(['nosplit', 'alsonosplit'])
    result = idx.str.rsplit('_', expand=True)
    exp = idx
    tm.assert_index_equal(result, exp)
    assert result.nlevels == 1