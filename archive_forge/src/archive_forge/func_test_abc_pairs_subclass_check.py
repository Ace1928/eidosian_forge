import re
import numpy as np
import pytest
from pandas.core.dtypes import generic as gt
import pandas as pd
import pandas._testing as tm
@pytest.mark.parametrize('abctype1, inst', abc_pairs)
@pytest.mark.parametrize('abctype2, _', abc_pairs)
def test_abc_pairs_subclass_check(self, abctype1, abctype2, inst, _):
    if abctype1 == abctype2:
        assert issubclass(type(inst), getattr(gt, abctype2))
        with pytest.raises(TypeError, match=re.escape('issubclass() arg 1 must be a class')):
            issubclass(inst, getattr(gt, abctype2))
    else:
        assert not issubclass(type(inst), getattr(gt, abctype2))