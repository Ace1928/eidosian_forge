from __future__ import annotations
import re
import warnings
from collections.abc import Iterable
import numpy as np
import pandas as pd
import pytest
from packaging.version import Version
import dask
import dask.dataframe as dd
from dask.dataframe._compat import PANDAS_GE_200, PANDAS_GE_300, tm
from dask.dataframe.core import apply_and_enforce
from dask.dataframe.utils import (
from dask.local import get_sync
def test_raise_on_meta_error():
    try:
        with raise_on_meta_error():
            raise RuntimeError('Bad stuff')
    except Exception as e:
        assert e.args[0].startswith('Metadata inference failed.\n')
        assert 'RuntimeError' in e.args[0]
    else:
        assert False, 'should have errored'
    try:
        with raise_on_meta_error('myfunc'):
            raise RuntimeError('Bad stuff')
    except Exception as e:
        assert e.args[0].startswith('Metadata inference failed in `myfunc`.\n')
        assert 'RuntimeError' in e.args[0]
    else:
        assert False, 'should have errored'