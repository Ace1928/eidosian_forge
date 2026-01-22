from ... import tests
from ...transport import memory
def test_cat_branch_revspec(self):
    wt = self.make_branch_and_tree('a')
    self.build_tree(['a/README'])
    wt.add('README')
    wt.commit('Making sure there is a basis_tree available')
    wt = self.make_branch_and_tree('b')
    out, err = self.run_bzr(['cat', '-r', 'branch:../a', 'README'], working_dir='b')
    self.assertEqual('contents of a/README\n', out)