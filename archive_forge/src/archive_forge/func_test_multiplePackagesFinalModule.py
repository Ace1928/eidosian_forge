import operator
import os
import shutil
import sys
import textwrap
import tempfile
from unittest import skipIf, TestCase
def test_multiplePackagesFinalModule(self):
    """
        L{wrapFQPN} returns a L{twisted.python.modules.PythonModule}
        referring to the deepest module described by dotted FQPN.
        """
    import xml.etree.ElementTree
    self.assertModuleWrapperRefersTo(self.wrapFQPN('xml.etree.ElementTree'), xml.etree.ElementTree)