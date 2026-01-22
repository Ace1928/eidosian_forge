import os
from typing import List
from .. import osutils, tests, win32utils
from ..win32utils import get_app_path, glob_expand
from . import TestCase, TestCaseInTempDir, TestSkipped, features
from .features import backslashdir_feature
def test_not_existing(self):
    p = get_app_path('not-existing')
    self.assertEqual('not-existing', p)