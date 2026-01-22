from breezy import commands, osutils, tests, trace, ui
from breezy.tests import script
def test_cat_bogus_output_file(self):
    self.run_script('\n$ cat >\n2>: No such file or directory\n')