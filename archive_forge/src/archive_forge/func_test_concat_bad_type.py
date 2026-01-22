import pytest
from pandas import (
from pandas.io.formats.style import Styler
def test_concat_bad_type(styler):
    msg = '`other` must be of type `Styler`'
    with pytest.raises(TypeError, match=msg):
        styler.concat(DataFrame([[1, 2]]))