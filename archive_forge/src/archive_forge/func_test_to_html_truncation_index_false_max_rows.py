from datetime import datetime
from io import StringIO
import itertools
import re
import textwrap
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
import pandas.io.formats.format as fmt
@pytest.mark.parametrize('index', [False, 0])
def test_to_html_truncation_index_false_max_rows(datapath, index):
    data = [[1.764052, 0.400157], [0.978738, 2.240893], [1.867558, -0.977278], [0.950088, -0.151357], [-0.103219, 0.410599]]
    df = DataFrame(data)
    result = df.to_html(max_rows=4, index=index)
    expected = expected_html(datapath, 'gh15019_expected_output')
    assert result == expected