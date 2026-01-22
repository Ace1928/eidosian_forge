from packaging.version import Version
import numpy as np
import skimage.data as data
from skimage.data._fetchers import _image_fetcher
from skimage import io
from skimage._shared.testing import assert_equal, assert_almost_equal, fetch
import os
import pytest
@pytest.mark.parametrize('function_name', ['file_hash'])
def test_fetchers_are_public(function_name):
    assert hasattr(data, function_name)