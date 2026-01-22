import datetime
from decimal import Decimal
from io import BytesIO
import os
import pathlib
import numpy as np
import pytest
from pandas._config import using_copy_on_write
from pandas._config.config import _get_option
from pandas.compat import is_platform_windows
from pandas.compat.pyarrow import (
import pandas as pd
import pandas._testing as tm
from pandas.util.version import Version
from pandas.io.parquet import (
def check_partition_names(path, expected):
    """Check partitions of a parquet file are as expected.

    Parameters
    ----------
    path: str
        Path of the dataset.
    expected: iterable of str
        Expected partition names.
    """
    import pyarrow.dataset as ds
    dataset = ds.dataset(path, partitioning='hive')
    assert dataset.partitioning.schema.names == expected