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
def test_filePathDeltaSiblingDir(self):
    """
        L{filePathDelta} can traverse upwards to create relative paths to
        siblings.
        """
    self.assertEqual(filePathDelta(FilePath('/foo/bar'), FilePath('/foo/baz')), ['..', 'baz'])