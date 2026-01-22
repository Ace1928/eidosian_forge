import re
import numpy as np
import pytest
from pandas.core.dtypes import generic as gt
import pandas as pd
import pandas._testing as tm
@pytest.mark.parametrize('parent, subs', abc_subclasses.items())
@pytest.mark.parametrize('abctype, inst', abc_pairs)
def test_abc_hierarchy(self, parent, subs, abctype, inst):
    if abctype in subs:
        assert isinstance(inst, getattr(gt, parent))
    else:
        assert not isinstance(inst, getattr(gt, parent))