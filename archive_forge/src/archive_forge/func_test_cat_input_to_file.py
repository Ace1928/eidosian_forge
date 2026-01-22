from breezy import commands, osutils, tests, trace, ui
from breezy.tests import script
def test_cat_input_to_file(self):
    retcode, out, err = self.run_command(['cat', '>file'], 'content\n', None, None)
    self.assertFileEqual('content\n', 'file')
    self.assertEqual(None, out)
    self.assertEqual(None, err)
    retcode, out, err = self.run_command(['cat', '>>file'], 'more\n', None, None)
    self.assertFileEqual('content\nmore\n', 'file')
    self.assertEqual(None, out)
    self.assertEqual(None, err)