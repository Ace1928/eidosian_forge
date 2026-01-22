from breezy import branch, controldir, errors, tests
from breezy.tests import script
def test_bind_when_bound(self):
    self.run_script('\n$ brz init trunk\n...\n$ brz init copy\n...\n$ cd copy\n$ brz bind ../trunk\n$ brz bind\n2>brz: ERROR: Branch is already bound\n')