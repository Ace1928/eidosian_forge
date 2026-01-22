import os
import platform
import shutil
import tempfile
import unittest
from gzip import GzipFile
from pathlib import Path
import pytest
from monty.shutil import (
@unittest.skipIf(platform.system() == 'Windows', 'Skip on windows')
def test_remove_folder(self):
    tempdir = tempfile.mkdtemp(dir=test_dir)
    remove(tempdir)
    assert not os.path.isdir(tempdir)