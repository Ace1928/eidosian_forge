import types
import testtools
from fixtures.callmany import CallMany
def test_exit_propagates_exceptions(self):
    call = CallMany()
    call.__enter__()
    self.assertEqual(False, call.__exit__(None, None, None))