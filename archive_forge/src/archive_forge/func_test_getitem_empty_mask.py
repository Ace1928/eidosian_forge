import os
import sys
import matplotlib
import numpy as np
import pandas
import pytest
from pandas._testing import ensure_clean
import modin.pandas as pd
from modin.config import MinPartitionSize, NPartitions, StorageFormat
from modin.pandas.indexing import is_range_like
from modin.pandas.testing import assert_index_equal
from modin.tests.pandas.utils import (
from modin.utils import get_current_execution
def test_getitem_empty_mask():
    modin_frames = []
    pandas_frames = []
    data1 = np.random.randint(0, 100, size=(100, 4))
    mdf1 = pd.DataFrame(data1, columns=list('ABCD'))
    pdf1 = pandas.DataFrame(data1, columns=list('ABCD'))
    modin_frames.append(mdf1)
    pandas_frames.append(pdf1)
    data2 = np.random.randint(0, 100, size=(100, 4))
    mdf2 = pd.DataFrame(data2, columns=list('ABCD'))
    pdf2 = pandas.DataFrame(data2, columns=list('ABCD'))
    modin_frames.append(mdf2)
    pandas_frames.append(pdf2)
    data3 = np.random.randint(0, 100, size=(100, 4))
    mdf3 = pd.DataFrame(data3, columns=list('ABCD'))
    pdf3 = pandas.DataFrame(data3, columns=list('ABCD'))
    modin_frames.append(mdf3)
    pandas_frames.append(pdf3)
    modin_data = pd.concat(modin_frames)
    pandas_data = pandas.concat(pandas_frames)
    df_equals(modin_data[[False for _ in modin_data.index]], pandas_data[[False for _ in modin_data.index]])