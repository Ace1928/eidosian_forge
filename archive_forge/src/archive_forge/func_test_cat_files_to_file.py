from breezy import commands, osutils, tests, trace, ui
from breezy.tests import script
def test_cat_files_to_file(self):
    self.build_tree_contents([('cat', b'cat\n')])
    self.build_tree_contents([('dog', b'dog\n')])
    retcode, out, err = self.run_command(['cat', 'cat', 'dog', '>file'], None, None, None)
    self.assertFileEqual(b'cat\ndog\n', 'file')