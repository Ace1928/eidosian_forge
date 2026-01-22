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
@pytest.mark.parametrize('size', [1, 5])
def test_html_invalid_formatters_arg_raises(size):
    df = DataFrame(columns=['a', 'b', 'c'])
    msg = 'Formatters length({}) should match DataFrame number of columns(3)'
    with pytest.raises(ValueError, match=re.escape(msg.format(size))):
        df.to_html(formatters=['{}'.format] * size)