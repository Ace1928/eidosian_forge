import os
import testtools
from testtools.matchers import StartsWith
from fixtures import (
def test_under_dir(self):
    root = self.useFixture(TempDir()).path
    fixture = TempHomeDir(root)
    fixture.setUp()
    with fixture:
        self.assertThat(fixture.path, StartsWith(root))