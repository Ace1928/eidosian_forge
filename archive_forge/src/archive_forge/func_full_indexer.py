import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
@pytest.fixture(params=[lambda x: x.index, lambda x: list(x.index), lambda x: slice(None), lambda x: slice(0, len(x)), lambda x: range(len(x)), lambda x: list(range(len(x))), lambda x: np.ones(len(x), dtype=bool)], ids=['index', 'list[index]', 'null_slice', 'full_slice', 'range', 'list(range)', 'mask'])
def full_indexer(self, request):
    """
        Fixture for an indexer to pass to obj.loc to get/set the full length of the
        object.

        In some cases, assumes that obj.index is the default RangeIndex.
        """
    return request.param