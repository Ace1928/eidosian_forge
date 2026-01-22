import os
import pytest
from wasabi.tables import row, table
from wasabi.util import supports_ansi
def test_colors_whole_table_supports_ansi_false(data, header, footer, fg_colors, bg_colors):
    os.environ['ANSI_COLORS_DISABLED'] = 'True'
    result = table(data, header=header, footer=footer, divider=True, fg_colors=fg_colors, bg_colors=bg_colors)
    assert result == '\nCOL A            COL B   COL 3     \n--------------   -----   ----------\nHello            World   12344342  \nThis is a test   World   1234      \n--------------   -----   ----------\n                         2030203.00\n'
    del os.environ['ANSI_COLORS_DISABLED']