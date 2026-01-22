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
@pytest.mark.parametrize('chunksize', (-1, 0, 'apple'))
def test_iterator_errors(datapath, chunksize):
    dta_file = datapath('io', 'data', 'stata', 'stata-dta-partially-labeled.dta')
    with pytest.raises(ValueError, match='chunksize must be a positive'):
        with StataReader(dta_file, chunksize=chunksize):
            pass