from collections.abc import Iterator
from io import StringIO
from pathlib import Path
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.io.json._json import JsonReader
def test_readjson_unicode(request, monkeypatch, engine):
    if engine == 'pyarrow':
        reason = "Pyarrow only supports a file path as an input and line delimited jsonand doesn't support chunksize parameter."
        request.applymarker(pytest.mark.xfail(reason=reason, raises=ValueError))
    with tm.ensure_clean('test.json') as path:
        monkeypatch.setattr('locale.getpreferredencoding', lambda do_setlocale: 'cp949')
        with open(path, 'w', encoding='utf-8') as f:
            f.write('{"£©µÀÆÖÞßéöÿ":["АБВГДабвгд가"]}')
        result = read_json(path, engine=engine)
        expected = DataFrame({'£©µÀÆÖÞßéöÿ': ['АБВГДабвгд가']})
        tm.assert_frame_equal(result, expected)