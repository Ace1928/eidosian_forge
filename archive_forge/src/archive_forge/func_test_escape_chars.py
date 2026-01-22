from .. import cmdline, tests
from .features import backslashdir_feature
def test_escape_chars(self):
    self.assertAsTokens([(False, 'foo\\bar')], 'foo\\bar')