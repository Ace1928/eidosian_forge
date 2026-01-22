import numpy as np
import pandas._config.config as cf
from pandas import (
def test_config_on(self):
    df = DataFrame({'A': [1, 2]})
    with cf.option_context('display.html.table_schema', True):
        result = df._repr_data_resource_()
    assert result is not None