import os
import shutil
import tempfile
import warnings
from functools import partial
from importlib import resources
from pathlib import Path
from pickle import dumps, loads
import numpy as np
import pytest
from sklearn.datasets import (
from sklearn.datasets._base import (
from sklearn.datasets.tests.test_common import check_as_frame
from sklearn.preprocessing import scale
from sklearn.utils import Bunch
@pytest.mark.parametrize('path_container', [None, Path, _DummyPath])
def test_data_home(path_container, data_home):
    if path_container is not None:
        data_home = path_container(data_home)
    data_home = get_data_home(data_home=data_home)
    assert data_home == data_home
    assert os.path.exists(data_home)
    if path_container is not None:
        data_home = path_container(data_home)
    clear_data_home(data_home=data_home)
    assert not os.path.exists(data_home)
    data_home = get_data_home(data_home=data_home)
    assert os.path.exists(data_home)