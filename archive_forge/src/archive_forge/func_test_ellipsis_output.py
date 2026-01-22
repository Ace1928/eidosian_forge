from breezy import commands, osutils, tests, trace, ui
from breezy.tests import script
def test_ellipsis_output(self):
    story = '\n$ cat\n<first line\n<second line\n<last line\nfirst line\n...\nlast line\n'
    self.run_script(story)
    story = '\n$ brz not-a-command\n2>..."not-a-command"\n'
    self.run_script(story)
    story = '\n$ brz branch not-a-branch\n2>brz: ERROR: Not a branch...not-a-branch/".\n'
    self.run_script(story)