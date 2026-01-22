from unittest import SkipTest
def test_completions_short_plot_long_style2(self):
    """Suggest corresponding plot options"""
    suggestions = OptsCompleter.line_completer('%%opts AnElement [test=1] BarElement style(', self.completions, self.compositor_defs)
    self.assertEqual(suggestions, ['styleoptC1=', 'styleoptC2='])