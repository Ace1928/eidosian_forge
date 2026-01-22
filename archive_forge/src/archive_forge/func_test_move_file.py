from breezy import commands, osutils, tests, trace, ui
from breezy.tests import script
def test_move_file(self):
    self.run_script('$ echo content >file')
    self.assertPathExists('file')
    self.run_script('$ mv file new_name')
    self.assertPathDoesNotExist('file')
    self.assertPathExists('new_name')