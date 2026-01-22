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
def test_value_labels_old_format(self, datapath):
    dpath = datapath('io', 'data', 'stata', 'S4_EDUC1.dta')
    with StataReader(dpath) as reader:
        assert reader.value_labels() == {}