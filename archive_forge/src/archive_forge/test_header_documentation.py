from collections import namedtuple
from io import StringIO
import numpy as np
import pytest
from pandas.errors import ParserError
from pandas import (
import pandas._testing as tm

Tests that the file header is properly handled or inferred
during parsing for all of the parsers defined in parsers.py
