import datetime
from decimal import Decimal
from io import BytesIO
import os
import pathlib
import numpy as np
import pytest
import pandas as pd
from pandas import read_orc
import pandas._testing as tm
from pandas.core.arrays import StringArray
import pyarrow as pa
def test_orc_writer_dtypes_not_supported(orc_writer_dtypes_not_supported):
    pytest.importorskip('pyarrow')
    msg = 'The dtype of one or more columns is not supported yet.'
    with pytest.raises(NotImplementedError, match=msg):
        orc_writer_dtypes_not_supported.to_orc()