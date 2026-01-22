import contextlib
from datetime import datetime
import io
import os
from pathlib import Path
import numpy as np
import pytest
from pandas.compat import IS64
from pandas.errors import EmptyDataError
import pandas.util._test_decorators as td
import pandas as pd
import pandas._testing as tm
from pandas.io.sas.sas7bdat import SAS7BDATReader
@pytest.mark.parametrize('test_file, override_offset, override_value, expected_msg', [('test2.sas7bdat', 65536 + 55229, 128 | 15, 'Out of bounds'), ('test2.sas7bdat', 65536 + 55229, 16, 'unknown control byte'), ('test3.sas7bdat', 118170, 184, 'Out of bounds')])
def test_rle_rdc_exceptions(datapath, test_file, override_offset, override_value, expected_msg):
    """Errors in RLE/RDC decompression should propagate."""
    with open(datapath('io', 'sas', 'data', test_file), 'rb') as fd:
        data = bytearray(fd.read())
    data[override_offset] = override_value
    with pytest.raises(Exception, match=expected_msg):
        pd.read_sas(io.BytesIO(data), format='sas7bdat')