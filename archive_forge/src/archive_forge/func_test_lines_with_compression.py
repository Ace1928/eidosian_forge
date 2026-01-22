from io import (
import pytest
import pandas.util._test_decorators as td
import pandas as pd
import pandas._testing as tm
def test_lines_with_compression(compression):
    with tm.ensure_clean() as path:
        df = pd.read_json(StringIO('{"a": [1, 2, 3], "b": [4, 5, 6]}'))
        df.to_json(path, orient='records', lines=True, compression=compression)
        roundtripped_df = pd.read_json(path, lines=True, compression=compression)
        tm.assert_frame_equal(df, roundtripped_df)