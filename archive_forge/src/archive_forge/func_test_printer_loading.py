import os
import re
import time
import pytest
from wasabi.printer import Printer
from wasabi.util import MESSAGES, NO_UTF8, supports_ansi
@pytest.mark.parametrize('hide_animation', [False, True])
def test_printer_loading(hide_animation):
    p = Printer(hide_animation=hide_animation)
    print('\n')
    with p.loading('Loading...'):
        time.sleep(1)
    p.good('Success!')
    with p.loading('Something else...'):
        time.sleep(2)
    p.good('Yo!')
    with p.loading('Loading...'):
        time.sleep(1)
    p.good('Success!')