import os
import pytest
from wasabi.tables import row, table
from wasabi.util import supports_ansi
def test_table_default(data):
    result = table(data)
    assert result == '\nHello            World   12344342\nThis is a test   World   1234    \n'