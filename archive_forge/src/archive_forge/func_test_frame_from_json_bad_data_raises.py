import datetime
from datetime import timedelta
from decimal import Decimal
from io import (
import json
import os
import sys
import time
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.compat import IS64
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics
from pandas.io.json import ujson_dumps
@pytest.mark.parametrize('data,msg,orient', [('{"key":b:a:d}', 'Expected object or value', 'columns'), ('{"columns":["A","B"],"index":["2","3"],"data":[[1.0,"1"],[2.0,"2"],[null,"3"]]}', '|'.join(['Length of values \\(3\\) does not match length of index \\(2\\)']), 'split'), ('{"columns":["A","B","C"],"index":["1","2","3"],"data":[[1.0,"1"],[2.0,"2"],[null,"3"]]}', '3 columns passed, passed data had 2 columns', 'split'), ('{"badkey":["A","B"],"index":["2","3"],"data":[[1.0,"1"],[2.0,"2"],[null,"3"]]}', 'unexpected key\\(s\\): badkey', 'split')])
def test_frame_from_json_bad_data_raises(self, data, msg, orient):
    with pytest.raises(ValueError, match=msg):
        read_json(StringIO(data), orient=orient)