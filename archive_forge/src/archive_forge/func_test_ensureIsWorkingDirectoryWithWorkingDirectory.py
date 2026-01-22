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
def test_ensureIsWorkingDirectoryWithWorkingDirectory(self):
    """
        Calling the C{ensureIsWorkingDirectory} VCS command's method on a valid
        working directory doesn't produce any error.
        """
    reposDir = self.makeRepository(self.tmpDir)
    self.assertIsNone(self.createCommand.ensureIsWorkingDirectory(reposDir))