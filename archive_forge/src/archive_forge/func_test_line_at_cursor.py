import pytest
from IPython.utils.tokenutil import token_at_cursor, line_at_cursor
def test_line_at_cursor():
    cell = ''
    line, offset = line_at_cursor(cell, cursor_pos=11)
    assert line == ''
    assert offset == 0
    cell = 'One\nTwo\n'
    line, offset = line_at_cursor(cell, cursor_pos=4)
    assert line == 'Two\n'
    assert offset == 4
    cell = 'pri\npri'
    line, offset = line_at_cursor(cell, cursor_pos=7)
    assert line == 'pri'
    assert offset == 4