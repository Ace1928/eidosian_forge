from datetime import (
from io import StringIO
import re
import sys
from textwrap import dedent
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas import (
import pandas._testing as tm
def test_to_string_line_width(self):
    df = DataFrame(123, index=range(10, 15), columns=range(30))
    lines = df.to_string(line_width=80)
    assert max((len(line) for line in lines.split('\n'))) == 80