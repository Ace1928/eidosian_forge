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
def test_statareader_warns_when_used_without_context(datapath):
    file_path = datapath('io', 'data', 'stata', 'stata-compat-118.dta')
    with tm.assert_produces_warning(ResourceWarning, match='without using a context manager'):
        sr = StataReader(file_path)
        sr.read()
    with tm.assert_produces_warning(FutureWarning, match='is not part of the public API'):
        sr.close()