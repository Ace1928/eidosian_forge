import os
from breezy import bedding, tests, workingtree
def test_missing_format_arg(self):
    wt = self._make_simple_branch('a')
    self.run_bzr(['branch', 'a', 'b'])
    wt.commit('third revision')
    wt.commit('fourth revision')
    missing = self.run_bzr(['missing', '--log-format', 'short'], retcode=1, working_dir='b')[0]
    self.assertEqual(8, len(missing.splitlines()))