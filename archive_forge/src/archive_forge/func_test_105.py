import bz2
import datetime as dt
from datetime import datetime
import gzip
import io
import os
import struct
import tarfile
import zipfile
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import CategoricalDtype
import pandas._testing as tm
from pandas.core.frame import (
from pandas.io.parsers import read_csv
from pandas.io.stata import (
def test_105(self, datapath):
    dpath = datapath('io', 'data', 'stata', 'S4_EDUC1.dta')
    df = read_stata(dpath)
    df0 = [[1, 1, 3, -2], [2, 1, 2, -2], [4, 1, 1, -2]]
    df0 = DataFrame(df0)
    df0.columns = ['clustnum', 'pri_schl', 'psch_num', 'psch_dis']
    df0['clustnum'] = df0['clustnum'].astype(np.int16)
    df0['pri_schl'] = df0['pri_schl'].astype(np.int8)
    df0['psch_num'] = df0['psch_num'].astype(np.int8)
    df0['psch_dis'] = df0['psch_dis'].astype(np.float32)
    tm.assert_frame_equal(df.head(3), df0)