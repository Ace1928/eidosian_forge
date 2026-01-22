import os
import numpy as np
import pytest
from xarray.core.indexing import MemoryCachedArray
from ...data import datasets, load_arviz_data
from ...rcparams import (
from ...stats import compare
from ..helpers import models  # pylint: disable=unused-import
def test_warning_rc_file(caplog):
    """Test invalid lines and duplicated keys log warnings and bad value raises error."""
    path = os.path.dirname(os.path.abspath(__file__))
    read_rcfile(os.path.join(path, '../test.rcparams'))
    records = caplog.records
    assert len(records) == 1
    assert records[0].levelname == 'WARNING'
    assert 'Duplicate key' in caplog.text