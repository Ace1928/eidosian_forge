import operator
import os
import shutil
import sys
import textwrap
import tempfile
from unittest import skipIf, TestCase
def makeImportable(self, path):
    sys.path.append(path)