import os
import smtplib
from breezy import gpg, merge_directive, tests, workingtree
def test_merge_user_r(self):
    """If the user supplies -r, an error is emitted"""
    self.prepare_merge_directive()
    self.tree1.commit('baz', rev_id=b'baz-id')
    md_text = self.run_bzr(['merge-directive', self.tree2.basedir, self.tree1.basedir, '--plain'])[0]
    self.build_tree_contents([('../directive', md_text)])
    os.chdir('../tree2')
    self.run_bzr_error(('Cannot use -r with merge directives or bundles',), 'merge -r 2 ../directive')