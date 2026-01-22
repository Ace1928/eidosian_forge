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
def test_ensureIsWorkingDirectoryWithNonWorkingDirectory(self):
    """
        Calling the C{ensureIsWorkingDirectory} VCS command's method on an
        invalid working directory raises a L{NotWorkingDirectory} exception.
        """
    self.assertRaises(NotWorkingDirectory, self.createCommand.ensureIsWorkingDirectory, self.tmpDir)