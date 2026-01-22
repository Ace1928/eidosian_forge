from breezy import commands, osutils, tests, trace, ui
from breezy.tests import script
def test_mkdir_usage(self):
    self.assertRaises(SyntaxError, self.run_script, '$ mkdir')
    self.assertRaises(SyntaxError, self.run_script, '$ mkdir foo bar')