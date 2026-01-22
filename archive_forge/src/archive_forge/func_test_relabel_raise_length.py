import numpy as np
import pytest
from pandas import (
from pandas.io.formats.style import Styler
from pandas.io.formats.style_render import _str_escape
@pytest.mark.parametrize('hide, labels', [(False, [1, 2]), (True, [1, 2, 3, 4])])
def test_relabel_raise_length(styler_multi, hide, labels):
    if hide:
        styler_multi.hide(axis=0, subset=[('X', 'x'), ('Y', 'y')])
    with pytest.raises(ValueError, match='``labels`` must be of length equal'):
        styler_multi.relabel_index(labels=labels)