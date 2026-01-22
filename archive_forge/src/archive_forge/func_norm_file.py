import os
import sys
from .. import osutils, urlutils
from ..errors import PathNotChild
from . import TestCase, TestCaseInTempDir, TestSkipped, features
def norm_file(expected, path):
    url = normalize_url(path)
    self.assertStartsWith(url, 'file:///')
    if sys.platform == 'win32':
        url = url[len('file:///C:'):]
    else:
        url = url[len('file://'):]
    self.assertEndsWith(url, expected)