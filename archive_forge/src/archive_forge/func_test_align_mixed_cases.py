import io
import numpy as np
import pytest
from pandas import (
@pytest.mark.parametrize('align, exp', [('left', [no_bar(), bar_to(80), bar_to(100)]), ('right', [bar_to(100), bar_from_to(80, 100), no_bar()]), ('mid', [bar_to(60), bar_from_to(60, 80), bar_from_to(60, 100)]), ('zero', [bar_to(50), bar_from_to(50, 66.66), bar_from_to(50, 83.33)]), ('mean', [bar_to(50), bar_from_to(50, 66.66), bar_from_to(50, 83.33)]), (-0.0, [bar_to(50), bar_from_to(50, 66.66), bar_from_to(50, 83.33)]), (np.nanmedian, [bar_to(50), no_bar(), bar_from_to(50, 62.5)])])
@pytest.mark.parametrize('nans', [True, False])
def test_align_mixed_cases(df_mix, align, exp, nans):
    expected = {(0, 0): exp[0], (1, 0): exp[1], (2, 0): exp[2]}
    if nans:
        df_mix.loc[3, :] = np.nan
        expected.update({(3, 0): no_bar()})
    result = df_mix.style.bar(align=align)._compute().ctx
    assert result == expected