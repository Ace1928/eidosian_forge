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
def test_hierarchical_attrs_columns(parser, planet_df):
    expected = '<?xml version=\'1.0\' encoding=\'utf-8\'?>\n<data>\n  <row location="inner" type="terrestrial" count_mass="4" sum_mass="11.81" mean_mass="2.95"/>\n  <row location="outer" type="gas giant" count_mass="2" sum_mass="2466.5" mean_mass="1233.25"/>\n  <row location="outer" type="ice giant" count_mass="2" sum_mass="189.23" mean_mass="94.61"/>\n  <row location="All" type="" count_mass="8" sum_mass="2667.54" mean_mass="333.44"/>\n</data>'
    pvt = planet_df.pivot_table(index=['location', 'type'], values='mass', aggfunc=['count', 'sum', 'mean'], margins=True).round(2)
    output = pvt.to_xml(attr_cols=list(pvt.reset_index().columns.values), parser=parser)
    output = equalize_decl(output)
    assert output == expected