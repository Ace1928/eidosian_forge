import os
from ssl import SSLError
from socket import timeout
from urllib.error import HTTPError, URLError
import numpy as np
from numpy.testing import assert_, assert_array_equal
import pytest
from statsmodels.datasets import get_rdataset, webuse, check_internet, utils
def test_webuse_pandas():
    from statsmodels.compat.pandas import assert_frame_equal
    from statsmodels.datasets import macrodata
    dta = macrodata.load_pandas().data
    base_gh = 'https://github.com/statsmodels/statsmodels/raw/main/statsmodels/datasets/macrodata/'
    internet_available = check_internet(base_gh)
    if not internet_available:
        pytest.skip('Unable to retrieve file - skipping test')
    try:
        res1 = webuse('macrodata', baseurl=base_gh)
    except IGNORED_EXCEPTIONS:
        pytest.skip('Failed with HTTP Error, these are random')
    res1 = res1.astype(float)
    assert_frame_equal(res1, dta.astype(float))