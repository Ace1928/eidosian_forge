import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.io.sas.sasreader import read_sas
def test1_index(self, file01):
    data_csv = pd.read_csv(file01.replace('.xpt', '.csv'))
    data_csv = data_csv.set_index('SEQN')
    numeric_as_float(data_csv)
    data = read_sas(file01, index='SEQN', format='xport')
    tm.assert_frame_equal(data, data_csv, check_index_type=False)
    with read_sas(file01, index='SEQN', format='xport', iterator=True) as reader:
        data = reader.read(10)
    tm.assert_frame_equal(data, data_csv.iloc[0:10, :], check_index_type=False)
    with read_sas(file01, index='SEQN', format='xport', chunksize=10) as reader:
        data = reader.get_chunk()
    tm.assert_frame_equal(data, data_csv.iloc[0:10, :], check_index_type=False)