from testtools.matchers import *
from ...tests import CapturedCall, TestCase
from ..smart.client import CallHookParams
from .matchers import *
def test_no_vfs_calls(self):
    calls = [self._make_call('Branch.get_config_file', [])]
    self.assertIs(None, ContainsNoVfsCalls().match(calls))