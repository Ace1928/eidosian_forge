import os
import shutil
import tarfile
import tempfile
from testtools import TestCase
from testtools.matchers import (
from testtools.matchers._filesystem import (
def test_matcher(self):
    tempdir = self.mkdtemp()
    filename = os.path.join(tempdir, 'foo')
    self.create_file(filename, 'Hello World!')
    self.assertThat(filename, FileContains(matcher=DocTestMatches('Hello World!')))