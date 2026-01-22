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
@pytest.mark.parametrize('allowed_extensions', (['.txt'], ['.txt', '.json']))
def test_load_files_allowed_extensions(tmp_path, allowed_extensions):
    """Check the behaviour of `allowed_extension` in `load_files`."""
    d = tmp_path / 'sub'
    d.mkdir()
    files = ('file1.txt', 'file2.json', 'file3.json', 'file4.md')
    paths = [d / f for f in files]
    for p in paths:
        p.write_bytes(b'hello')
    res = load_files(tmp_path, allowed_extensions=allowed_extensions)
    assert set([str(p) for p in paths if p.suffix in allowed_extensions]) == set(res.filenames)