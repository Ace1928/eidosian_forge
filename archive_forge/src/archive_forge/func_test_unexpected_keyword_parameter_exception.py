import codecs
import csv
from io import StringIO
import os
from pathlib import Path
import numpy as np
import pytest
from pandas.compat import PY311
from pandas.errors import (
from pandas import DataFrame
import pandas._testing as tm
def test_unexpected_keyword_parameter_exception(all_parsers):
    parser = all_parsers
    msg = "{}\\(\\) got an unexpected keyword argument 'foo'"
    with pytest.raises(TypeError, match=msg.format('read_csv')):
        parser.read_csv('foo.csv', foo=1)
    with pytest.raises(TypeError, match=msg.format('read_table')):
        parser.read_table('foo.tsv', foo=1)