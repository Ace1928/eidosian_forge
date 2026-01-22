import operator
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.computation.check import NUMEXPR_INSTALLED
def test_lots_of_operators_string(self, df):
    res = df.query("`  &^ :!€$?(} >    <++*''  ` > 4")
    expect = df[df["  &^ :!€$?(} >    <++*''  "] > 4]
    tm.assert_frame_equal(res, expect)