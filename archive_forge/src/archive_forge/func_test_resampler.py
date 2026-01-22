import io
import warnings
import matplotlib
import numpy as np
import pandas
import pandas._libs.lib as lib
import pytest
from numpy.testing import assert_array_equal
import modin.pandas as pd
from modin.config import Engine, NPartitions, StorageFormat
from modin.pandas.io import to_pandas
from modin.tests.pandas.utils import (
from modin.tests.test_utils import warns_that_defaulting_to_pandas
from modin.utils import get_current_execution
@pytest.mark.parametrize('rule', ['5min', pandas.offsets.Hour()])
@pytest.mark.parametrize('axis', [0])
def test_resampler(rule, axis):
    data, index = (test_data_resample['data'], test_data_resample['index'])
    modin_resampler = pd.DataFrame(data, index=index).resample(rule, axis=axis)
    pandas_resampler = pandas.DataFrame(data, index=index).resample(rule, axis=axis)
    assert pandas_resampler.indices == modin_resampler.indices
    assert pandas_resampler.groups == modin_resampler.groups
    df_equals(modin_resampler.get_group(name=list(modin_resampler.groups)[0]), pandas_resampler.get_group(name=list(pandas_resampler.groups)[0]))