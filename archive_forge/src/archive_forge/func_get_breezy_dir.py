import ast
import os
import re
import sys
import breezy.branch
from breezy import osutils
from breezy.tests import TestCase, TestSkipped, features
def get_breezy_dir(self):
    """Get the path to the root of breezy"""
    source = self.source_file_name(breezy)
    source_dir = os.path.dirname(source)
    if not os.path.isdir(source_dir):
        raise TestSkipped('Cannot find breezy source directory. Expected %s' % source_dir)
    return source_dir