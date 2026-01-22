from io import (
import pytest
import pandas.util._test_decorators as td
import pandas as pd
import pandas._testing as tm
def test_chunksize_with_compression(compression):
    with tm.ensure_clean() as path:
        df = pd.read_json(StringIO('{"a": ["foo", "bar", "baz"], "b": [4, 5, 6]}'))
        df.to_json(path, orient='records', lines=True, compression=compression)
        with pd.read_json(path, lines=True, chunksize=1, compression=compression) as res:
            roundtripped_df = pd.concat(res)
        tm.assert_frame_equal(df, roundtripped_df)