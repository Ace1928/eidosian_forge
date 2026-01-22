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
def test_to_html_pos_args_deprecation():
    df = DataFrame({'a': [1, 2, 3]})
    msg = "Starting with pandas version 3.0 all arguments of to_html except for the argument 'buf' will be keyword-only."
    with tm.assert_produces_warning(FutureWarning, match=msg):
        df.to_html(None, None)