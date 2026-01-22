import os
import shutil
import tarfile
import tempfile
from testtools import TestCase
from testtools.matchers import (
from testtools.matchers._filesystem import (
def test_does_not_contain(self):
    tempdir = self.mkdtemp()
    filename = os.path.join(tempdir, 'foo')
    self.create_file(filename, 'Goodbye Cruel World!')
    mismatch = FileContains('Hello World!').match(filename)
    self.assertThat(Equals('Hello World!').match('Goodbye Cruel World!').describe(), Equals(mismatch.describe()))