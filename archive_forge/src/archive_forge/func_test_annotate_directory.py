from breezy import config, tests
from breezy.urlutils import joinpath
from ..test_bedding import override_whoami
def test_annotate_directory(self):
    """Test --directory option"""
    wt = self.make_branch_and_tree('a')
    self.build_tree_contents([('a/hello.txt', b'my helicopter\n')])
    wt.add(['hello.txt'])
    wt.commit('commit', committer='test@user')
    out, err = self.run_bzr(['annotate', '-d', 'a', 'hello.txt'])
    self.assertEqualDiff('1   test@us | my helicopter\n', out)