from io import (
import os
import platform
from urllib.error import URLError
import uuid
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
def test_memory_map_compression(all_parsers, compression):
    """
    Support memory map for compressed files.

    GH 37621
    """
    parser = all_parsers
    expected = DataFrame({'a': [1], 'b': [2]})
    with tm.ensure_clean() as path:
        expected.to_csv(path, index=False, compression=compression)
        if parser.engine == 'pyarrow':
            msg = "The 'memory_map' option is not supported with the 'pyarrow' engine"
            with pytest.raises(ValueError, match=msg):
                parser.read_csv(path, memory_map=True, compression=compression)
            return
        result = parser.read_csv(path, memory_map=True, compression=compression)
    tm.assert_frame_equal(result, expected)