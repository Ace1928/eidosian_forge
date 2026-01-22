import sys
import types
from testtools import TestCase
from testtools.matchers import (
from extras import (
def test_error_callback_not_on_success(self):
    check_error_callback(self, try_import, 'os.path', 0, True)