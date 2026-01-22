import pytest
import pandas as pd
from pandas import MultiIndex
import pandas._testing as tm
def test_multiindex_name_and_level_raising():
    mi = MultiIndex.from_arrays([[1, 2], [3, 4]], names=['x', 'y'])
    with pytest.raises(TypeError, match='Can not pass level for dictlike `names`.'):
        mi.set_names(names={'x': 'z'}, level={'x': 'z'})