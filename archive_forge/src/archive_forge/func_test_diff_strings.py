import pytest
from wasabi.util import color, diff_strings, format_repr, locale_escape, wrap
def test_diff_strings():
    a = 'hello\nworld\nwide\nweb'
    b = 'yo\nwide\nworld\nweb'
    expected = '\x1b[38;5;16;48;5;2myo\x1b[0m\n\x1b[38;5;16;48;5;2mwide\x1b[0m\n\x1b[38;5;16;48;5;1mhello\x1b[0m\nworld\n\x1b[38;5;16;48;5;1mwide\x1b[0m\nweb'
    assert diff_strings(a, b) == expected