from __future__ import annotations
import csv
from io import (
from typing import TYPE_CHECKING
import numpy as np
import pytest
from pandas.errors import (
from pandas import (
import pandas._testing as tm
def test_python_engine_file_no_next(python_parser_only):
    parser = python_parser_only

    class NoNextBuffer:

        def __init__(self, csv_data) -> None:
            self.data = csv_data

        def __iter__(self) -> Iterator:
            return self.data.__iter__()

        def read(self):
            return self.data

        def readline(self):
            return self.data
    parser.read_csv(NoNextBuffer('a\n1'))