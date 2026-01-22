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
def test___repr__does_not_raise_attribute_column_warning():
    df = pd.DataFrame([1])
    with warnings.catch_warnings():
        warnings.filterwarnings(action='error', message=SET_DATAFRAME_ATTRIBUTE_WARNING)
        repr(df)