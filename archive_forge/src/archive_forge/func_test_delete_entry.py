from io import BytesIO
from dulwich.repo import Repo as GitRepo
from .. import tests
def test_delete_entry(self):
    stream = BytesIO()
    builder = tests.GitBranchBuilder(stream)
    builder.delete_entry('path/to/fµ')
    self.assertEqual([b'D path/to/f\xc2\xb5\n'], builder.commit_info)