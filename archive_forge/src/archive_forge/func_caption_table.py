import codecs
from datetime import datetime
from textwrap import dedent
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.fixture
def caption_table(self):
    """Caption for table/tabular LaTeX environment."""
    return 'a table in a \\texttt{table/tabular} environment'