from unittest import SkipTest
def test_completions_short_norm1(self):
    """Suggest corresponding plot options"""
    suggestions = OptsCompleter.line_completer('%%opts AnElement {', self.completions, self.compositor_defs)
    self.assertEqual(suggestions, ['+axiswise', '+framewise'])