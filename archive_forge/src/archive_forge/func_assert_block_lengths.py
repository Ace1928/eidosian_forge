import numpy as np
import pytest
from pandas._libs import lib
import pandas.util._test_decorators as td
import pandas as pd
import pandas._testing as tm
def assert_block_lengths(x):
    assert len(x) == len(x._mgr.blocks[0].mgr_locs)
    return 0