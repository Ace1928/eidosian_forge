from packaging.version import Version
import numpy as np
import skimage.data as data
from skimage.data._fetchers import _image_fetcher
from skimage import io
from skimage._shared.testing import assert_equal, assert_almost_equal, fetch
import os
import pytest
def test_download_all_with_pooch():
    data_dir = data.data_dir
    if _image_fetcher is not None:
        data.download_all()
        assert 'astronaut.png' in os.listdir(data_dir)
        assert len(os.listdir(data_dir)) > 50
    else:
        with pytest.raises(ModuleNotFoundError):
            data.download_all()