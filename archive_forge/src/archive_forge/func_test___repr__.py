import io
import warnings
import matplotlib
import numpy as np
import pandas
import pytest
import modin.pandas as pd
from modin.config import NPartitions
from modin.pandas.utils import SET_DATAFRAME_ATTRIBUTE_WARNING
from modin.tests.pandas.utils import (
from modin.tests.test_utils import warns_that_defaulting_to_pandas
def test___repr__():
    frame_data = random_state.randint(RAND_LOW, RAND_HIGH, size=(1000, 100))
    pandas_df = pandas.DataFrame(frame_data)
    modin_df = pd.DataFrame(frame_data)
    assert repr(pandas_df) == repr(modin_df)
    frame_data = random_state.randint(RAND_LOW, RAND_HIGH, size=(1000, 99))
    pandas_df = pandas.DataFrame(frame_data)
    modin_df = pd.DataFrame(frame_data)
    assert repr(pandas_df) == repr(modin_df)
    frame_data = random_state.randint(RAND_LOW, RAND_HIGH, size=(1000, 101))
    pandas_df = pandas.DataFrame(frame_data)
    modin_df = pd.DataFrame(frame_data)
    assert repr(pandas_df) == repr(modin_df)
    frame_data = random_state.randint(RAND_LOW, RAND_HIGH, size=(1000, 102))
    pandas_df = pandas.DataFrame(frame_data)
    modin_df = pd.DataFrame(frame_data)
    assert repr(pandas_df) == repr(modin_df)
    frame_data = random_state.randint(RAND_LOW, RAND_HIGH, size=(10, 100))
    pandas_df = pandas.DataFrame(frame_data)
    modin_df = pd.DataFrame(frame_data)
    assert repr(pandas_df) == repr(modin_df)
    frame_data = random_state.randint(RAND_LOW, RAND_HIGH, size=(10, 10))
    pandas_df = pandas.DataFrame(frame_data)
    modin_df = pd.DataFrame(frame_data)
    assert repr(pandas_df) == repr(modin_df)
    frame_data = random_state.randint(RAND_LOW, RAND_HIGH, size=(100, 10))
    pandas_df = pandas.DataFrame(frame_data)
    modin_df = pd.DataFrame(frame_data)
    assert repr(pandas_df) == repr(modin_df)
    pandas_df = pandas.DataFrame(columns=['col{}'.format(i) for i in range(100)])
    modin_df = pd.DataFrame(columns=['col{}'.format(i) for i in range(100)])
    assert repr(pandas_df) == repr(modin_df)
    string_data = '"time","device_id","lat","lng","accuracy","activity_1","activity_1_conf","activity_2","activity_2_conf","activity_3","activity_3_conf"\n"2016-08-26 09:00:00.206",2,60.186805,24.821049,33.6080017089844,"STILL",75,"IN_VEHICLE",5,"ON_BICYCLE",5\n"2016-08-26 09:00:05.428",5,60.192928,24.767222,5,"WALKING",62,"ON_BICYCLE",29,"RUNNING",6\n"2016-08-26 09:00:05.818",1,60.166382,24.700443,3,"WALKING",75,"IN_VEHICLE",5,"ON_BICYCLE",5\n"2016-08-26 09:00:15.816",1,60.166254,24.700671,3,"WALKING",75,"IN_VEHICLE",5,"ON_BICYCLE",5\n"2016-08-26 09:00:16.413",5,60.193055,24.767427,5,"WALKING",85,"ON_BICYCLE",15,"UNKNOWN",0\n"2016-08-26 09:00:20.578",3,60.152996,24.745216,3.90000009536743,"STILL",69,"IN_VEHICLE",31,"UNKNOWN",0'
    pandas_df = pandas.read_csv(io.StringIO(string_data))
    with warns_that_defaulting_to_pandas():
        modin_df = pd.read_csv(io.StringIO(string_data))
    assert repr(pandas_df) == repr(modin_df)