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
def test_load_files_w_categories_desc_and_encoding(test_category_dir_1, test_category_dir_2, load_files_root):
    category = os.path.abspath(test_category_dir_1).split(os.sep).pop()
    res = load_files(load_files_root, description='test', categories=[category], encoding='utf-8')
    assert len(res.filenames) == 1
    assert len(res.target_names) == 1
    assert res.DESCR == 'test'
    assert res.data == ['Hello World!\n']