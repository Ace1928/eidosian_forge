import codecs
from datetime import datetime
from textwrap import dedent
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.fixture
def label_longtable(self):
    """Label for longtable LaTeX environment."""
    return 'tab:longtable'