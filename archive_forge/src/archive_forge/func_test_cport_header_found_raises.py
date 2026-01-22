import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.io.sas.sasreader import read_sas
def test_cport_header_found_raises(self, file05):
    msg = 'Header record indicates a CPORT file, which is not readable.'
    with pytest.raises(ValueError, match=msg):
        read_sas(file05, format='xport')