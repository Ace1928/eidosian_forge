from io import BytesIO
from dulwich.repo import Repo as GitRepo
from .. import tests
def test_add_and_commit(self):
    stream = BytesIO()
    builder = tests.GitBranchBuilder(stream)
    builder.set_file('fµ/bar', b'contents\nbar\n', False)
    self.assertEqual(b'2', builder.commit(b'Joe Foo <joe@foo.com>', 'committing fµ/bar', timestamp=1194586400, timezone=b'+0100'))
    self.assertEqualDiff(b'blob\nmark :1\ndata 13\ncontents\nbar\n\ncommit refs/heads/master\nmark :2\ncommitter Joe Foo <joe@foo.com> 1194586400 +0100\ndata 18\ncommitting f\xc2\xb5/bar\nM 100644 :1 f\xc2\xb5/bar\n\n', stream.getvalue())