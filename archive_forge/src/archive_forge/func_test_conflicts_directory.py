from breezy import conflicts, tests, workingtree
from breezy.tests import features, script
def test_conflicts_directory(self):
    self.run_script('$ brz conflicts  -d branch\nText conflict in my_other_file\nPath conflict: mydir3 / mydir2\nText conflict in myfile\n')