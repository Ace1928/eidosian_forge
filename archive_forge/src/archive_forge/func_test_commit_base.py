from io import BytesIO
from dulwich.repo import Repo as GitRepo
from .. import tests
def test_commit_base(self):
    stream = BytesIO()
    builder = tests.GitBranchBuilder(stream)
    builder.set_file('foo', b'contents\nfoo\n', False)
    r1 = builder.commit(b'Joe Foo <joe@foo.com>', 'first', timestamp=1194586400)
    r2 = builder.commit(b'Joe Foo <joe@foo.com>', 'second', timestamp=1194586405)
    r3 = builder.commit(b'Joe Foo <joe@foo.com>', 'third', timestamp=1194586410, base=r1)
    self.assertEqualDiff(b'blob\nmark :1\ndata 13\ncontents\nfoo\n\ncommit refs/heads/master\nmark :2\ncommitter Joe Foo <joe@foo.com> 1194586400 +0000\ndata 5\nfirst\nM 100644 :1 foo\n\ncommit refs/heads/master\nmark :3\ncommitter Joe Foo <joe@foo.com> 1194586405 +0000\ndata 6\nsecond\n\ncommit refs/heads/master\nmark :4\ncommitter Joe Foo <joe@foo.com> 1194586410 +0000\ndata 5\nthird\nfrom :2\n\n', stream.getvalue())