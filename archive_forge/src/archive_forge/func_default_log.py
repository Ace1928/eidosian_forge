import os
import sys
import breezy
from breezy import osutils, trace
from breezy.tests import (TestCase, TestCaseInTempDir, TestSkipped,
def default_log(self):
    return os.path.join(os.environ['BRZ_HOME'], 'breezy', 'brz.log')