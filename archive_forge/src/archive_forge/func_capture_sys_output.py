import pytest
import sys
from contextlib import contextmanager
from io import StringIO
from ...utils import nipype_cmd
@contextmanager
def capture_sys_output():
    caputure_out, capture_err = (StringIO(), StringIO())
    current_out, current_err = (sys.stdout, sys.stderr)
    try:
        sys.stdout, sys.stderr = (caputure_out, capture_err)
        yield (caputure_out, capture_err)
    finally:
        sys.stdout, sys.stderr = (current_out, current_err)