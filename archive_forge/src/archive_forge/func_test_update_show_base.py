import os
from breezy import branch, osutils, tests, workingtree
from breezy.bzr import bzrdir
from breezy.tests.script import ScriptRunner
def test_update_show_base(self):
    """brz update support --show-base

        see https://bugs.launchpad.net/bzr/+bug/202374"""
    tree = self.make_branch_and_tree('.')
    with open('hello', 'w') as f:
        f.write('foo')
    tree.add('hello')
    tree.commit('fie')
    with open('hello', 'w') as f:
        f.write('fee')
    tree.commit('fee')
    self.run_bzr(['update', '-r1'])
    with open('hello', 'w') as f:
        f.write('fie')
    out, err = self.run_bzr(['update', '--show-base'], retcode=1)
    self.assertContainsString(err, ' M  hello\nText conflict in hello\n1 conflicts encountered.\n')
    with open('hello', 'rb') as f:
        self.assertEqualDiff(b'<<<<<<< TREE\nfie||||||| BASE-REVISION\nfoo=======\nfee>>>>>>> MERGE-SOURCE\n', f.read())