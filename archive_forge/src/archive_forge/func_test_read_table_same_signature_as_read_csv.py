from datetime import datetime
from inspect import signature
from io import StringIO
import os
from pathlib import Path
import sys
import numpy as np
import pytest
from pandas.errors import (
from pandas import (
import pandas._testing as tm
from pandas.io.parsers import TextFileReader
from pandas.io.parsers.c_parser_wrapper import CParserWrapper
def test_read_table_same_signature_as_read_csv(all_parsers):
    parser = all_parsers
    table_sign = signature(parser.read_table)
    csv_sign = signature(parser.read_csv)
    assert table_sign.parameters.keys() == csv_sign.parameters.keys()
    assert table_sign.return_annotation == csv_sign.return_annotation
    for key, csv_param in csv_sign.parameters.items():
        table_param = table_sign.parameters[key]
        if key == 'sep':
            assert csv_param.default == ','
            assert table_param.default == '\t'
            assert table_param.annotation == csv_param.annotation
            assert table_param.kind == csv_param.kind
            continue
        assert table_param == csv_param