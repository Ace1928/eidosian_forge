import glob
import operator
import os
import shutil
import sys
import tempfile
from incremental import Version
from twisted.python import release
from twisted.python._release import (
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
def test_statusNotClean(self):
    """
        Calling the C{isStatusClean} VCS command's method on a repository with
        no pending modifications returns C{False}.
        """
    reposDir = self.makeRepository(self.tmpDir)
    reposDir.child('some-file').setContent(b'something')
    self.assertFalse(self.createCommand.isStatusClean(reposDir))