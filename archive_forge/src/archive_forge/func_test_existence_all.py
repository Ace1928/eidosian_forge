from scipy.datasets._registry import registry
from scipy.datasets._fetchers import data_fetcher
from scipy.datasets._utils import _clear_cache
from scipy.datasets import ascent, face, electrocardiogram, download_all
from numpy.testing import assert_equal, assert_almost_equal
import os
import pytest
def test_existence_all(self):
    assert len(os.listdir(data_dir)) >= len(registry)