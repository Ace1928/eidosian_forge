from datetime import (
from functools import partial
from io import BytesIO
import os
import re
import numpy as np
import pytest
from pandas.compat import is_platform_windows
from pandas.compat._constants import PY310
from pandas.compat._optional import import_optional_dependency
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.io.excel import (
from pandas.io.excel._util import _writers
def test_register_writer(self):

    class DummyClass(ExcelWriter):
        called_save = False
        called_write_cells = False
        called_sheets = False
        _supported_extensions = ('xlsx', 'xls')
        _engine = 'dummy'

        def book(self):
            pass

        def _save(self):
            type(self).called_save = True

        def _write_cells(self, *args, **kwargs):
            type(self).called_write_cells = True

        @property
        def sheets(self):
            type(self).called_sheets = True

        @classmethod
        def assert_called_and_reset(cls):
            assert cls.called_save
            assert cls.called_write_cells
            assert not cls.called_sheets
            cls.called_save = False
            cls.called_write_cells = False
    register_writer(DummyClass)
    with option_context('io.excel.xlsx.writer', 'dummy'):
        path = 'something.xlsx'
        with tm.ensure_clean(path) as filepath:
            with ExcelWriter(filepath) as writer:
                assert isinstance(writer, DummyClass)
            df = DataFrame(['a'], columns=Index(['b'], name='foo'), index=Index(['c'], name='bar'))
            df.to_excel(filepath)
        DummyClass.assert_called_and_reset()
    with tm.ensure_clean('something.xls') as filepath:
        df.to_excel(filepath, engine='dummy')
    DummyClass.assert_called_and_reset()