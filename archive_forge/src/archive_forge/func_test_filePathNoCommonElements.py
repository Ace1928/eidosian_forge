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
def test_filePathNoCommonElements(self):
    """
        L{filePathDelta} can create relative paths to totally unrelated paths
        for maximum portability.
        """
    self.assertEqual(filePathDelta(FilePath('/foo/bar'), FilePath('/baz/quux')), ['..', '..', 'baz', 'quux'])