import codecs
from datetime import datetime
from textwrap import dedent
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.fixture
def short_caption(self):
    """Short caption for testing \\caption[short_caption]{full_caption}."""
    return 'a table'