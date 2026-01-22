from __future__ import annotations
from io import (
import os
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.io.common import get_handle
from pandas.io.xml import read_xml
@td.skip_if_installed('lxml')
def test_default_parser_no_lxml(geom_df):
    with pytest.raises(ImportError, match='lxml not found, please install or use the etree parser.'):
        geom_df.to_xml()