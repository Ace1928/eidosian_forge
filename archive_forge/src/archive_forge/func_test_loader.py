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
@pytest.mark.parametrize('loader_func, data_shape, target_shape, n_target, has_descr, filenames', [(load_breast_cancer, (569, 30), (569,), 2, True, ['filename']), (load_wine, (178, 13), (178,), 3, True, []), (load_iris, (150, 4), (150,), 3, True, ['filename']), (load_linnerud, (20, 3), (20, 3), 3, True, ['data_filename', 'target_filename']), (load_diabetes, (442, 10), (442,), None, True, []), (load_digits, (1797, 64), (1797,), 10, True, []), (partial(load_digits, n_class=9), (1617, 64), (1617,), 10, True, [])])
def test_loader(loader_func, data_shape, target_shape, n_target, has_descr, filenames):
    bunch = loader_func()
    assert isinstance(bunch, Bunch)
    assert bunch.data.shape == data_shape
    assert bunch.target.shape == target_shape
    if hasattr(bunch, 'feature_names'):
        assert len(bunch.feature_names) == data_shape[1]
    if n_target is not None:
        assert len(bunch.target_names) == n_target
    if has_descr:
        assert bunch.DESCR
    if filenames:
        assert 'data_module' in bunch
        assert all([f in bunch and (resources.files(bunch['data_module']) / bunch[f]).is_file() for f in filenames])