from unittest import SkipTest
def test_completions_invalid_plot1(self):
    """Same as above although the syntax is invalid"""
    suggestions = OptsCompleter.line_completer('%%opts Ano [', self.completions, self.compositor_defs)
    self.assertEqual(suggestions, self.all_keys)