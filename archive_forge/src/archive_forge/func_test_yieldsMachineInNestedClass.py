import operator
import os
import shutil
import sys
import textwrap
import tempfile
from unittest import skipIf, TestCase
def test_yieldsMachineInNestedClass(self):
    """
        When given a L{twisted.python.modules.PythonAttribute} that refers
        to a nested class that contains a L{MethodicalMachine} as a
        class variable, L{findMachinesViaWrapper} yields that machine
        and its FQPN.
        """
    source = '        from automat import MethodicalMachine\n\n        class PythonClass(object):\n            class NestedClass(object):\n                _classMachine = MethodicalMachine()\n        '
    moduleDict = self.makeModuleAsDict(source, self.pathDir, 'nestedcls.py')
    PythonClass = moduleDict['nestedcls.PythonClass']
    self.assertIn(('nestedcls.PythonClass.NestedClass._classMachine', PythonClass.load().NestedClass._classMachine), list(self.findMachinesViaWrapper(PythonClass)))