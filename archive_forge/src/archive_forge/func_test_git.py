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
def test_git(self):
    """
        L{GitCommand} implements L{IVCSCommand}.
        """
    self.assertTrue(IVCSCommand.implementedBy(GitCommand))