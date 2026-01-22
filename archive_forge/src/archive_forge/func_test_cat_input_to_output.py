from breezy import commands, osutils, tests, trace, ui
from breezy.tests import script
def test_cat_input_to_output(self):
    retcode, out, err = self.run_command(['cat'], 'content\n', 'content\n', None)
    self.assertEqual('content\n', out)
    self.assertEqual(None, err)