import re
import numpy as np
import pytest
from pandas.core.dtypes import generic as gt
import pandas as pd
import pandas._testing as tm
@pytest.mark.parametrize('abctype1, inst', abc_pairs)
@pytest.mark.parametrize('abctype2, _', abc_pairs)
def test_abc_pairs_instance_check(self, abctype1, abctype2, inst, _):
    if abctype1 == abctype2:
        assert isinstance(inst, getattr(gt, abctype2))
        assert not isinstance(type(inst), getattr(gt, abctype2))
    else:
        assert not isinstance(inst, getattr(gt, abctype2))