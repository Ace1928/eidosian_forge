import io
import numpy as np
import pytest
from pandas import (
@pytest.mark.parametrize('align, exp', [('left', [bar_to(100), bar_to(50), no_bar()]), ('right', [no_bar(), bar_from_to(50, 100), bar_to(100)]), ('mid', [bar_from_to(66.66, 100), bar_from_to(33.33, 100), bar_to(100)]), ('zero', [bar_from_to(33.33, 50), bar_from_to(16.66, 50), bar_to(50)]), ('mean', [bar_from_to(50, 100), no_bar(), bar_to(50)]), (-2.0, [bar_from_to(50, 100), no_bar(), bar_to(50)]), (np.median, [bar_from_to(50, 100), no_bar(), bar_to(50)])])
def test_align_negative_cases(df_neg, align, exp):
    result = df_neg.style.bar(align=align)._compute().ctx
    expected = {(0, 0): exp[0], (1, 0): exp[1], (2, 0): exp[2]}
    assert result == expected