import operator
import os
import shutil
import sys
import textwrap
import tempfile
from unittest import skipIf, TestCase
def makeModule(self, source, path, moduleName):
    pythonModuleName, _ = os.path.splitext(moduleName)
    return self.writeSourceInto(source, path, moduleName)[pythonModuleName]