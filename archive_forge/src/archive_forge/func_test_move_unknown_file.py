from breezy import commands, osutils, tests, trace, ui
from breezy.tests import script
def test_move_unknown_file(self):
    self.assertRaises(AssertionError, self.run_script, '$ mv unknown does-not-exist')