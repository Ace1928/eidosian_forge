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
@pytest.fixture(scope='module')
def data_home(tmpdir_factory):
    tmp_file = str(tmpdir_factory.mktemp('scikit_learn_data_home_test'))
    yield tmp_file
    _remove_dir(tmp_file)