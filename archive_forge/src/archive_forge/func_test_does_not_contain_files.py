import os
import shutil
import tarfile
import tempfile
from testtools import TestCase
from testtools.matchers import (
from testtools.matchers._filesystem import (
def test_does_not_contain_files(self):
    tempdir = self.mkdtemp()
    self.touch(os.path.join(tempdir, 'foo'))
    mismatch = DirContains(['bar', 'foo']).match(tempdir)
    self.assertThat(Equals(['bar', 'foo']).match(['foo']).describe(), Equals(mismatch.describe()))