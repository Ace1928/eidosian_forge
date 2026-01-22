import os
import pytest
from wasabi.tables import row, table
from wasabi.util import supports_ansi
def test_row_fg_colors(fg_colors):
    result = row(('Hello', 'World', '12344342'), fg_colors=fg_colors)
    if SUPPORTS_ANSI:
        assert result == 'Hello   \x1b[38;5;3mWorld\x1b[0m   \x1b[38;5;87m12344342\x1b[0m'
    else:
        assert result == 'Hello   World   12344342'