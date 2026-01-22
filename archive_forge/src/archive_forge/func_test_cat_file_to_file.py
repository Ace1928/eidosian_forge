from breezy import commands, osutils, tests, trace, ui
from breezy.tests import script
def test_cat_file_to_file(self):
    self.build_tree_contents([('file', b'content\n')])
    retcode, out, err = self.run_command(['cat', 'file', '>file2'], None, None, None)
    self.assertFileEqual(b'content\n', 'file2')