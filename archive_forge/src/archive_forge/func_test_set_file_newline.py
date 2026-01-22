from io import BytesIO
from dulwich.repo import Repo as GitRepo
from .. import tests
def test_set_file_newline(self):
    stream = BytesIO()
    builder = tests.GitBranchBuilder(stream)
    builder.set_file('foo\nbar', b'contents\nbar\n', False)
    self.assertEqualDiff(b'blob\nmark :1\ndata 13\ncontents\nbar\n\n', stream.getvalue())
    self.assertEqual([b'M 100644 :1 "foo\\nbar"\n'], builder.commit_info)