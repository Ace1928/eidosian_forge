import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_astype_invalid_conversion(self):
    df = DataFrame({'a': [1, 2, 'text'], 'b': [1, 2, 3]})
    msg = "invalid literal for int() with base 10: 'text': Error while type casting for column 'a'"
    with pytest.raises(ValueError, match=re.escape(msg)):
        df.astype({'a': int})