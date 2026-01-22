import os
import re
import time
import pytest
from wasabi.printer import Printer
from wasabi.util import MESSAGES, NO_UTF8, supports_ansi
def test_color_and_bc_color():
    p = Printer(no_print=True)
    text = 'This is a text.'
    result = p.text(text, color='green', bg_color='yellow')
    print(result)
    if SUPPORTS_ANSI:
        assert result == '\x1b[38;5;2;48;5;3mThis is a text.\x1b[0m'
    else:
        assert result == 'This is a text.'