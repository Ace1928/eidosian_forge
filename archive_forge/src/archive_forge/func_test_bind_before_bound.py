from breezy import branch, controldir, errors, tests
from breezy.tests import script
def test_bind_before_bound(self):
    self.run_script('\n$ brz init trunk\n...\n$ cd trunk\n$ brz bind\n2>brz: ERROR: No location supplied and no previous location known\n')