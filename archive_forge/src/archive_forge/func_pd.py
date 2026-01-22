import pytest
import sys
import matplotlib
from matplotlib import _api
@pytest.fixture
def pd():
    """Fixture to import and configure pandas."""
    pd = pytest.importorskip('pandas')
    try:
        from pandas.plotting import deregister_matplotlib_converters as deregister
        deregister()
    except ImportError:
        pass
    return pd