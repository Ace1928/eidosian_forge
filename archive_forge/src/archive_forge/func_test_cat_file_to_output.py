from breezy import commands, osutils, tests, trace, ui
from breezy.tests import script
def test_cat_file_to_output(self):
    self.build_tree_contents([('file', b'content\n')])
    retcode, out, err = self.run_command(['cat', 'file'], None, 'content\n', None)
    self.assertEqual('content\n', out)
    self.assertEqual(None, err)