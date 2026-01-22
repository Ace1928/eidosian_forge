from breezy import commands, osutils, tests, trace, ui
from breezy.tests import script
def test_cd_usage(self):
    self.assertRaises(SyntaxError, self.run_script, '$ cd foo bar')