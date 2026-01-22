import codecs
from datetime import datetime
from textwrap import dedent
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_to_latex_column_format_just_works(self, float_frame):
    float_frame.to_latex(column_format='lcr')