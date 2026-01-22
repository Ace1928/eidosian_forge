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
@pytest.mark.parametrize('version', [114, 117])
def test_invalid_variable_label_encoding(self, version, mixed_frame):
    mixed_frame.index.name = 'index'
    variable_labels = {'a': 'very long' * 10, 'b': 'City Exponent', 'c': 'City'}
    variable_labels['a'] = 'invalid character Œ'
    with tm.ensure_clean() as path:
        with pytest.raises(ValueError, match='Variable labels must contain only characters'):
            mixed_frame.to_stata(path, variable_labels=variable_labels, version=version)