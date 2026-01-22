from breezy import commands, osutils, tests, trace, ui
from breezy.tests import script
def test_echo_some_to_output(self):
    retcode, out, err = self.run_command(['echo', 'hello'], None, 'hello\n', None)
    self.assertEqual('hello\n', out)
    self.assertEqual(None, err)