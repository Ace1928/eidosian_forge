import os
import pytest
from wasabi.tables import row, table
from wasabi.util import supports_ansi
def test_row_bg_colors(bg_colors):
    result = row(('Hello', 'World', '12344342'), bg_colors=bg_colors)
    if SUPPORTS_ANSI:
        assert result == '\x1b[48;5;2mHello\x1b[0m   \x1b[48;5;23mWorld\x1b[0m   12344342'
    else:
        assert result == 'Hello   World   12344342'