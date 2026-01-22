import ast
import os
import re
import sys
import breezy.branch
from breezy import osutils
from breezy.tests import TestCase, TestSkipped, features
def is_license_exception(self, fname):
    """Certain files are allowed to be different"""
    if not self.is_our_code(fname):
        return True
    for exc in LICENSE_EXCEPTIONS:
        if fname.endswith(exc):
            return True
    return False