from .. import cmdline, tests
from .features import backslashdir_feature
def test_nested_quotations(self):
    self.assertAsTokens([(True, 'foo"" bar')], '"foo\\"\\" bar"')
    self.assertAsTokens([(True, "foo'' bar")], '"foo\'\' bar"')
    self.assertAsTokens([(True, "foo'' bar")], '"foo\'\' bar"', single_quotes_allowed=True)
    self.assertAsTokens([(True, 'foo"" bar')], '\'foo"" bar\'', single_quotes_allowed=True)