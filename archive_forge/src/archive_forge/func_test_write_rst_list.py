import os
import time
from pathlib import Path
from unittest import mock, SkipTest
import pytest
from ...testing import TempFATFS
from ...utils.filemanip import (
@pytest.mark.parametrize('items,expected', [('', ' \n\n'), ('A string', ' A string\n\n'), (['A list', 'Of strings'], ' A list\n Of strings\n\n'), (None, TypeError)])
def test_write_rst_list(tmp_path, items, expected):
    if items is not None:
        assert write_rst_list(items) == expected
    else:
        with pytest.raises(expected):
            write_rst_list(items)