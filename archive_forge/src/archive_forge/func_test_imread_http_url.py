import os
import pathlib
import tempfile
import numpy as np
import pytest
from skimage import io
from skimage._shared.testing import assert_array_equal, fetch
from skimage.data import data_dir
def test_imread_http_url(httpserver):
    httpserver.serve_content(one_by_one_jpeg)
    image = io.imread(httpserver.url + '/test.jpg' + '?' + 's' * 266)
    assert image.shape == (1, 1)