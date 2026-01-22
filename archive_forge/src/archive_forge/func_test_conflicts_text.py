from breezy import conflicts, tests, workingtree
from breezy.tests import features, script
def test_conflicts_text(self):
    self.run_script('$ cd branch\n$ brz conflicts --text\nmy_other_file\nmyfile\n')