import os
from typing import List
from .. import osutils, tests, win32utils
from ..win32utils import get_app_path, glob_expand
from . import TestCase, TestCaseInTempDir, TestSkipped, features
from .features import backslashdir_feature
def build_unicode_tree(self):
    self.requireFeature(features.UnicodeFilenameFeature)
    self.build_tree(['ሴ', 'ሴሴ', 'ስ/', 'ስ/ስ'])