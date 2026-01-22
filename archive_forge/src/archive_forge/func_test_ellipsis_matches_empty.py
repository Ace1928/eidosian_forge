from breezy import commands, osutils, tests, trace, ui
from breezy.tests import script
def test_ellipsis_matches_empty(self):
    self.run_script('\n        $ cd .\n        ...\n        ')