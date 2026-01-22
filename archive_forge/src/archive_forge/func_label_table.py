import codecs
from datetime import datetime
from textwrap import dedent
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.fixture
def label_table(self):
    """Label for table/tabular LaTeX environment."""
    return 'tab:table_tabular'