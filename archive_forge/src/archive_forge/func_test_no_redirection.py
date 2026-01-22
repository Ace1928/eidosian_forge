from breezy import commands, osutils, tests, trace, ui
from breezy.tests import script
def test_no_redirection(self):
    self._check(None, None, None, [], [])
    self._check(None, None, None, ['foo', 'bar'], ['foo', 'bar'])