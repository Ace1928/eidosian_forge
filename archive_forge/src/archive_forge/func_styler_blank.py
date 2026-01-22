import gc
import numpy as np
import pytest
from pandas import (
import matplotlib as mpl
from pandas.io.formats.style import Styler
@pytest.fixture
def styler_blank(df_blank):
    return Styler(df_blank, uuid_len=0)