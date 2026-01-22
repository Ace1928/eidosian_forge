from breezy import commands, osutils, tests, trace, ui
from breezy.tests import script
def test_output_redirection(self):
    self._check(None, 'foo', 'w+', [], ['>foo'])
    self._check(None, 'foo', 'w+', ['bar'], ['bar', '>foo'])
    self._check(None, 'foo', 'w+', ['bar'], ['bar', '>', 'foo'])
    self._check(None, 'foo', 'a+', [], ['>>foo'])
    self._check(None, 'foo', 'a+', ['bar'], ['bar', '>>foo'])
    self._check(None, 'foo', 'a+', ['bar'], ['bar', '>>', 'foo'])