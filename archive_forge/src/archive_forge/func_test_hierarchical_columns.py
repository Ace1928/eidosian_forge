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
def test_hierarchical_columns(parser, planet_df):
    expected = "<?xml version='1.0' encoding='utf-8'?>\n<data>\n  <row>\n    <location>inner</location>\n    <type>terrestrial</type>\n    <count_mass>4</count_mass>\n    <sum_mass>11.81</sum_mass>\n    <mean_mass>2.95</mean_mass>\n  </row>\n  <row>\n    <location>outer</location>\n    <type>gas giant</type>\n    <count_mass>2</count_mass>\n    <sum_mass>2466.5</sum_mass>\n    <mean_mass>1233.25</mean_mass>\n  </row>\n  <row>\n    <location>outer</location>\n    <type>ice giant</type>\n    <count_mass>2</count_mass>\n    <sum_mass>189.23</sum_mass>\n    <mean_mass>94.61</mean_mass>\n  </row>\n  <row>\n    <location>All</location>\n    <type/>\n    <count_mass>8</count_mass>\n    <sum_mass>2667.54</sum_mass>\n    <mean_mass>333.44</mean_mass>\n  </row>\n</data>"
    pvt = planet_df.pivot_table(index=['location', 'type'], values='mass', aggfunc=['count', 'sum', 'mean'], margins=True).round(2)
    output = pvt.to_xml(parser=parser)
    output = equalize_decl(output)
    assert output == expected