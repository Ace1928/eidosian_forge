from datetime import (
import re
import numpy as np
import pytest
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.concat import concat
from pandas.core.reshape.merge import (
def test_mergeerror_on_left_index_mismatched_dtypes():
    df_1 = DataFrame(data=['X'], columns=['C'], index=[22])
    df_2 = DataFrame(data=['X'], columns=['C'], index=[999])
    with pytest.raises(MergeError, match='Can only pass argument'):
        merge(df_1, df_2, on=['C'], left_index=True)