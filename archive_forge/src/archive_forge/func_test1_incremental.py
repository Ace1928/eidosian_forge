import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.io.sas.sasreader import read_sas
def test1_incremental(self, file01):
    data_csv = pd.read_csv(file01.replace('.xpt', '.csv'))
    data_csv = data_csv.set_index('SEQN')
    numeric_as_float(data_csv)
    with read_sas(file01, index='SEQN', chunksize=1000) as reader:
        all_data = list(reader)
    data = pd.concat(all_data, axis=0)
    tm.assert_frame_equal(data, data_csv, check_index_type=False)