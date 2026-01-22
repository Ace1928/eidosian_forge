import numpy as np
import pytest
from pandas._libs.internals import BlockPlacement
import pandas.util._test_decorators as td
import pandas as pd
from pandas.core.internals import BlockManager
from pandas.core.internals.blocks import ExtensionBlock
def test_concat_axis1(df):
    df2 = pd.DataFrame({'c': [0.1, 0.2, 0.3]})
    res = pd.concat([df, df2], axis=1)
    assert isinstance(res._mgr.blocks[1], CustomBlock)