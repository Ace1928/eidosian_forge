from contextlib import contextmanager
from io import StringIO
import sys
import os
@contextmanager
def pycharm():
    os.environ['PYCHARM_HOSTED'] = '1'
    non_tty = StreamNonTTY()
    with replace_by(non_tty), replace_original_by(non_tty):
        yield
    del os.environ['PYCHARM_HOSTED']