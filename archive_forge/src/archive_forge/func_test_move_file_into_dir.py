from breezy import commands, osutils, tests, trace, ui
from breezy.tests import script
def test_move_file_into_dir(self):
    self.run_script('\n$ mkdir dir\n$ echo content > file\n')
    self.run_script('$ mv file dir')
    self.assertPathExists('dir')
    self.assertPathDoesNotExist('file')
    self.assertPathExists('dir/file')