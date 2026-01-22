import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.io.sas.sasreader import read_sas
@pytest.fixture
def file04(self, datapath):
    return datapath('io', 'sas', 'data', 'paxraw_d_short.xpt')