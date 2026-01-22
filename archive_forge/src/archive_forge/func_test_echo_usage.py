from breezy import commands, osutils, tests, trace, ui
from breezy.tests import script
def test_echo_usage(self):
    story = '\n$ echo foo\n<bar\n'
    self.assertRaises(SyntaxError, self.run_script, story)