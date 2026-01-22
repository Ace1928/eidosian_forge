from breezy import commands, osutils, tests, trace, ui
from breezy.tests import script
def test_command_with_error(self):
    story = '\n$ brz branch foo\n2>brz: ERROR: Not a branch: "foo"\n'
    self.assertEqual([(['brz', 'branch', 'foo'], None, None, 'brz: ERROR: Not a branch: "foo"\n')], script._script_to_commands(story))