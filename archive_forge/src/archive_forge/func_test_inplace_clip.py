import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_inplace_clip(self, float_frame):
    median = float_frame.median().median()
    frame_copy = float_frame.copy()
    return_value = frame_copy.clip(upper=median, lower=median, inplace=True)
    assert return_value is None
    assert not (frame_copy.values != median).any()