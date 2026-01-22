import os
from breezy import bedding, tests, workingtree
def test_missing_default_format(self):
    wt = self._make_simple_branch('a')
    self.run_bzr(['branch', 'a', 'b'])
    wt.commit('third revision')
    wt.commit('fourth revision')
    missing = self.run_bzr('missing', retcode=1, working_dir='b')[0]
    self.assertEqual(4, len(missing.splitlines()))