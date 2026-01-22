import os
import re
import time
import pytest
from wasabi.printer import Printer
from wasabi.util import MESSAGES, NO_UTF8, supports_ansi
def test_printer_print_timestamp():
    p = Printer(no_print=True, timestamp=True)
    result = p.info('Hello world')
    matches = re.match('^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}', result)
    assert matches