from functools import partial
import pytest
from sklearn.datasets.tests.test_common import check_return_X_y
def test_fetch_asframe(fetch_california_housing_fxt):
    pd = pytest.importorskip('pandas')
    bunch = fetch_california_housing_fxt(as_frame=True)
    frame = bunch.frame
    assert hasattr(bunch, 'frame') is True
    assert frame.shape == (20640, 9)
    assert isinstance(bunch.data, pd.DataFrame)
    assert isinstance(bunch.target, pd.Series)