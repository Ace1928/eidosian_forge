import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
def convert_nested_indexer(indexer_type, keys):
    if indexer_type == np.ndarray:
        return np.array(keys)
    if indexer_type == slice:
        return slice(*keys)
    return indexer_type(keys)