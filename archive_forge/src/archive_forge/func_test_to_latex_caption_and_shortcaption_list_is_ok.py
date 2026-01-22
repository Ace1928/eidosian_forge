import codecs
from datetime import datetime
from textwrap import dedent
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_to_latex_caption_and_shortcaption_list_is_ok(self, df_short):
    caption = ('Long-long-caption', 'Short')
    result_tuple = df_short.to_latex(caption=caption)
    result_list = df_short.to_latex(caption=list(caption))
    assert result_tuple == result_list