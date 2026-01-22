from statsmodels.compat.python import lrange
from io import BytesIO
from itertools import product
import numpy as np
from numpy.testing import assert_, assert_raises
import pandas as pd
import pytest
from statsmodels.api import datasets
from statsmodels.graphics.mosaicplot import (
@pytest.mark.smoke
@pytest.mark.matplotlib
def test_mosaic_empty_cells(close_figures):
    import pandas as pd
    mydata = pd.DataFrame({'id2': {64: 'Angelica', 65: 'DXW_UID', 66: 'casuid01', 67: 'casuid01', 68: 'EC93_uid', 69: 'EC93_uid', 70: 'EC93_uid', 60: 'DXW_UID', 61: 'AtmosFox', 62: 'DXW_UID', 63: 'DXW_UID'}, 'id1': {64: 'TGP', 65: 'Retention01', 66: 'default', 67: 'default', 68: 'Musa_EC_9_3', 69: 'Musa_EC_9_3', 70: 'Musa_EC_9_3', 60: 'default', 61: 'default', 62: 'default', 63: 'default'}})
    ct = pd.crosstab(mydata.id1, mydata.id2)
    _, vals = mosaic(ct.T.unstack())
    _, vals = mosaic(mydata, ['id1', 'id2'])