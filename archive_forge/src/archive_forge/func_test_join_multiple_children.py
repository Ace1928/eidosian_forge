import os
import tempfile
import testtools
from testtools.matchers import StartsWith
from fixtures import (
def test_join_multiple_children(self):
    temp_dir = self.useFixture(TempDir())
    root = temp_dir.path
    self.assertEqual(os.path.join(root, 'foo', 'bar', 'baz'), temp_dir.join('foo', 'bar', 'baz'))