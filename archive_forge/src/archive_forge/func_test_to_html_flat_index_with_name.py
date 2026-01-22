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
def test_to_html_flat_index_with_name(self, datapath, df, expected_without_index):
    df.index = Index(['foo', 'bar', 'baz'], name='idx')
    expected_with_index = expected_html(datapath, 'index_3')
    assert df.to_html() == expected_with_index
    assert df.to_html(index=False) == expected_without_index