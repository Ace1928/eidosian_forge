from datetime import datetime
import itertools
import re
import numpy as np
import pytest
from pandas._libs import lib
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape import reshape as reshape_lib
@pytest.mark.slow
def test_unstack_number_of_levels_larger_than_int32(self, monkeypatch):

    class MockUnstacker(reshape_lib._Unstacker):

        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            raise Exception("Don't compute final result.")
    with monkeypatch.context() as m:
        m.setattr(reshape_lib, '_Unstacker', MockUnstacker)
        df = DataFrame(np.zeros((2 ** 16, 2)), index=[np.arange(2 ** 16), np.arange(2 ** 16)])
        msg = 'The following operation may generate'
        with tm.assert_produces_warning(PerformanceWarning, match=msg):
            with pytest.raises(Exception, match="Don't compute final result."):
                df.unstack()