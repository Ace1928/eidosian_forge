import operator
import re
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
@pytest.mark.filterwarnings("ignore:DataFrame.fillna with 'method' is deprecated:FutureWarning", 'ignore:last is deprecated:FutureWarning')
def test_finalize_called(ndframe_method):
    cls, init_args, method = ndframe_method
    ndframe = cls(*init_args)
    ndframe.attrs = {'a': 1}
    result = method(ndframe)
    assert result.attrs == {'a': 1}