from unittest import SkipTest
def test_completions_simple1(self):
    suggestions = OptsCompleter.line_completer('%%opts An', self.completions, self.compositor_defs)
    self.assertEqual(suggestions, self.all_keys)