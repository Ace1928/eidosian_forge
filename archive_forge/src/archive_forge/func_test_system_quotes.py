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
def test_system_quotes(self):
    status = system('%s -c "import sys"' % python)
    self.assertEqual(status, 0)