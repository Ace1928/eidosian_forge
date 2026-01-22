import os
import pytest
from wasabi.tables import row, table
from wasabi.util import supports_ansi
def test_row_single_widths():
    data = ('a', 'bb', 'ccc')
    result = row(data, widths=10)
    assert result == 'a            bb           ccc       '