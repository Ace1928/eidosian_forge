import codecs
from datetime import datetime
from textwrap import dedent
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.fixture
def caption_longtable(self):
    """Caption for longtable LaTeX environment."""
    return 'a table in a \\texttt{longtable} environment'