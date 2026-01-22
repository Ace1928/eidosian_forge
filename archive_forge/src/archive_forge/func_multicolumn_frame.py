import codecs
from datetime import datetime
from textwrap import dedent
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.fixture
def multicolumn_frame(self):
    """Multicolumn dataframe for testing multicolumn LaTeX macros."""
    yield DataFrame({('c1', 0): {x: x for x in range(5)}, ('c1', 1): {x: x + 5 for x in range(5)}, ('c2', 0): {x: x for x in range(5)}, ('c2', 1): {x: x + 5 for x in range(5)}, ('c3', 0): {x: x for x in range(5)}})