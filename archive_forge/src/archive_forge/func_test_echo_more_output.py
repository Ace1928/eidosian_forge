from breezy import commands, osutils, tests, trace, ui
from breezy.tests import script
def test_echo_more_output(self):
    retcode, out, err = self.run_command(['echo', 'hello', 'happy', 'world'], None, 'hello happy world\n', None)
    self.assertEqual('hello happy world\n', out)
    self.assertEqual(None, err)