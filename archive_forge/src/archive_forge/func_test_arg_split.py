import sys
import signal
import os
import time
from _thread import interrupt_main  # Py 3
import threading
import pytest
from IPython.utils.process import (find_cmd, FindCmdError, arg_split,
from IPython.utils.capture import capture_output
from IPython.testing import decorators as dec
from IPython.testing import tools as tt
@dec.skip_win32
@pytest.mark.parametrize('argstr, argv', [('hi', ['hi']), ('hello there', ['hello', 'there']), ('hǎllo', ['hǎllo']), ('something "with quotes"', ['something', '"with quotes"'])])
def test_arg_split(argstr, argv):
    """Ensure that argument lines are correctly split like in a shell."""
    assert arg_split(argstr) == argv