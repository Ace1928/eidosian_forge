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
def test_no_pretty_print_no_decl(parser, geom_df):
    expected = '<data><row><index>0</index><shape>square</shape><degrees>360</degrees><sides>4.0</sides></row><row><index>1</index><shape>circle</shape><degrees>360</degrees><sides/></row><row><index>2</index><shape>triangle</shape><degrees>180</degrees><sides>3.0</sides></row></data>'
    output = geom_df.to_xml(xml_declaration=False, pretty_print=False, parser=parser)
    if output is not None:
        output = output.replace(' />', '/>')
    assert output == expected