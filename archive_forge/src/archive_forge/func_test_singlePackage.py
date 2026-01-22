import operator
import os
import shutil
import sys
import textwrap
import tempfile
from unittest import skipIf, TestCase
def test_singlePackage(self):
    """
        L{wrapFQPN} returns a L{twisted.python.modules.PythonModule}
        referring to the single package a dotless FQPN describes.
        """
    import xml
    self.assertModuleWrapperRefersTo(self.wrapFQPN('xml'), xml)