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
def test_find_cmd_fail():
    """Make sure that FindCmdError is raised if we can't find the cmd."""
    pytest.raises(FindCmdError, find_cmd, 'asdfasdf')