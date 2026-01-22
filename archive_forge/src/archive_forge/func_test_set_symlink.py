from io import BytesIO
from dulwich.repo import Repo as GitRepo
from .. import tests
def test_set_symlink(self):
    stream = BytesIO()
    builder = tests.GitBranchBuilder(stream)
    builder.set_symlink('fµ/bar', b'link/contents')
    self.assertEqualDiff(b'blob\nmark :1\ndata 13\nlink/contents\n', stream.getvalue())
    self.assertEqual([b'M 120000 :1 f\xc2\xb5/bar\n'], builder.commit_info)