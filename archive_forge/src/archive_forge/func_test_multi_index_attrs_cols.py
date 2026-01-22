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
def test_multi_index_attrs_cols(parser, planet_df):
    expected = '<?xml version=\'1.0\' encoding=\'utf-8\'?>\n<data>\n  <row location="inner" type="terrestrial" count="4" sum="11.81" mean="2.95"/>\n  <row location="outer" type="gas giant" count="2" sum="2466.5" mean="1233.25"/>\n  <row location="outer" type="ice giant" count="2" sum="189.23" mean="94.61"/>\n</data>'
    agg = planet_df.groupby(['location', 'type'])['mass'].agg(['count', 'sum', 'mean']).round(2)
    output = agg.to_xml(attr_cols=list(agg.reset_index().columns.values), parser=parser)
    output = equalize_decl(output)
    assert output == expected