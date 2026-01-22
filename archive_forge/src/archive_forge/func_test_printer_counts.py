import os
import re
import time
import pytest
from wasabi.printer import Printer
from wasabi.util import MESSAGES, NO_UTF8, supports_ansi
def test_printer_counts():
    p = Printer()
    text = 'This is a test.'
    for i in range(2):
        p.good(text)
    for i in range(1):
        p.fail(text)
    for i in range(4):
        p.warn(text)
    assert p.counts[MESSAGES.GOOD] == 2
    assert p.counts[MESSAGES.FAIL] == 1
    assert p.counts[MESSAGES.WARN] == 4