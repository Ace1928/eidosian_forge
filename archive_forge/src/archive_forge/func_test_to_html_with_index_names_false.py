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
def test_to_html_with_index_names_false():
    df = DataFrame({'A': [1, 2]}, index=Index(['a', 'b'], name='myindexname'))
    result = df.to_html(index_names=False)
    assert 'myindexname' not in result