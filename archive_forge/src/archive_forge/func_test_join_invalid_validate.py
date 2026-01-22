from datetime import datetime
import numpy as np
import pytest
from pandas.errors import MergeError
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.concat import concat
def test_join_invalid_validate(left_no_dup, right_no_dup):
    msg = '"invalid" is not a valid argument. Valid arguments are:\n- "1:1"\n- "1:m"\n- "m:1"\n- "m:m"\n- "one_to_one"\n- "one_to_many"\n- "many_to_one"\n- "many_to_many"'
    with pytest.raises(ValueError, match=msg):
        left_no_dup.merge(right_no_dup, on='a', validate='invalid')