import numpy as np
import pytest
from pandas import (
def test_maybe_infer_to_datetimelike_ser_construct():
    result = Series(['M1701', Timestamp('20130101')])
    assert result.dtype.kind == 'O'