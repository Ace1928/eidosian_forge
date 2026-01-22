import pytest
from thinc.api import (
from thinc.compat import has_tensorflow, has_torch
@pytest.fixture(scope='module')
def mnist(limit=5000):
    pytest.importorskip('ml_datasets')
    import ml_datasets
    (train_X, train_Y), (dev_X, dev_Y) = ml_datasets.mnist()
    return ((train_X[:limit], train_Y[:limit]), (dev_X[:limit], dev_Y[:limit]))