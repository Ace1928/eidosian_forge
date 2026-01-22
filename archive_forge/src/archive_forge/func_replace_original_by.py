from contextlib import contextmanager
from io import StringIO
import sys
import os
@contextmanager
def replace_original_by(stream):
    orig_stdout = sys.__stdout__
    orig_stderr = sys.__stderr__
    sys.__stdout__ = stream
    sys.__stderr__ = stream
    yield
    sys.__stdout__ = orig_stdout
    sys.__stderr__ = orig_stderr