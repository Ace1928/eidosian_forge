import pytest
from datashader.datashape import (
def test_no_promote_option():
    x = int64
    y = Option(float64)
    z = promote(x, y, promote_option=False)
    assert z == float64