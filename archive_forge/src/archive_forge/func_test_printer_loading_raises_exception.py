import os
import re
import time
import pytest
from wasabi.printer import Printer
from wasabi.util import MESSAGES, NO_UTF8, supports_ansi
def test_printer_loading_raises_exception():

    def loading_with_exception():
        p = Printer()
        print('\n')
        with p.loading():
            raise Exception('This is an error.')
    with pytest.raises(Exception):
        loading_with_exception()