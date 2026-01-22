import codecs
from datetime import datetime
from textwrap import dedent
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_to_latex_escape_default(self, df_with_symbols):
    default = df_with_symbols.to_latex()
    specified_true = df_with_symbols.to_latex(escape=True)
    assert default != specified_true