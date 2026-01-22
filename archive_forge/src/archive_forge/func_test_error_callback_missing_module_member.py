import sys
import types
from testtools import TestCase
from testtools.matchers import (
from extras import (
def test_error_callback_missing_module_member(self):
    check_error_callback(self, try_import, 'os.nonexistent', 1, False)