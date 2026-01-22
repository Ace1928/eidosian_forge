from breezy import commands, osutils, tests, trace, ui
from breezy.tests import script
def test_echo_appended(self):
    retcode, out, err = self.run_command(['echo', 'hello', '>file'], None, None, None)
    self.assertEqual(None, out)
    self.assertEqual(None, err)
    self.assertFileEqual(b'hello\n', 'file')
    retcode, out, err = self.run_command(['echo', 'happy', '>>file'], None, None, None)
    self.assertEqual(None, out)
    self.assertEqual(None, err)
    self.assertFileEqual(b'hello\nhappy\n', 'file')