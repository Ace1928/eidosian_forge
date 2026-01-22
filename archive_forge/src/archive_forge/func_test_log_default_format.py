import os
from breezy import bedding, tests, workingtree
def test_log_default_format(self):
    self._make_simple_branch()
    log = self.run_bzr('log')[0]
    self.assertEqual(2, len(log.splitlines()))