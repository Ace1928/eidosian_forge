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
def test_get_output_error_code(self):
    quiet_exit = '%s -c "import sys; sys.exit(1)"' % python
    out, err, code = get_output_error_code(quiet_exit)
    self.assertEqual(out, '')
    self.assertEqual(err, '')
    self.assertEqual(code, 1)
    out, err, code = get_output_error_code(f'{python} "{self.fname}"')
    self.assertEqual(out, 'on stdout')
    self.assertEqual(err, 'on stderr')
    self.assertEqual(code, 0)