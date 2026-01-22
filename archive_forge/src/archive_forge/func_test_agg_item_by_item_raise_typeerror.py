import datetime as dt
from functools import partial
import numpy as np
import pytest
from pandas.errors import SpecificationError
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.io.formats.printing import pprint_thing
def test_agg_item_by_item_raise_typeerror():
    df = DataFrame(np.random.default_rng(2).integers(10, size=(20, 10)))

    def raiseException(df):
        pprint_thing('----------------------------------------')
        pprint_thing(df.to_string())
        raise TypeError('test')
    with pytest.raises(TypeError, match='test'):
        df.groupby(0).agg(raiseException)