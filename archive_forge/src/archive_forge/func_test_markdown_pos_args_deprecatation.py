from io import (
import pytest
import pandas as pd
import pandas._testing as tm
def test_markdown_pos_args_deprecatation():
    df = pd.DataFrame({'a': [1, 2, 3]})
    msg = "Starting with pandas version 3.0 all arguments of to_markdown except for the argument 'buf' will be keyword-only."
    with tm.assert_produces_warning(FutureWarning, match=msg):
        buffer = BytesIO()
        df.to_markdown(buffer, 'grid')