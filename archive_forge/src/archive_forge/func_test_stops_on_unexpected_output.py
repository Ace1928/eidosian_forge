from breezy import commands, osutils, tests, trace, ui
from breezy.tests import script
def test_stops_on_unexpected_output(self):
    story = '\n$ mkdir dir\n$ cd dir\nThe cd command ouputs nothing\n'
    self.assertRaises(AssertionError, self.run_script, story)