from breezy import commands, osutils, tests, trace, ui
from breezy.tests import script
def test_mkdir_jailed(self):
    self.assertRaises(ValueError, self.run_script, '$ mkdir /out-of-jail')
    self.assertRaises(ValueError, self.run_script, '$ mkdir ../out-of-jail')