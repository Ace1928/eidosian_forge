import operator
import os
import shutil
import sys
import textwrap
import tempfile
from unittest import skipIf, TestCase
def test_yieldsMachineInClass(self):
    """
        When given a L{twisted.python.modules.PythonAttribute} that refers
        to a class that contains a L{MethodicalMachine} as a class
        variable, L{findMachinesViaWrapper} yields that machine and
        its FQPN.
        """
    source = '        from automat import MethodicalMachine\n\n        class PythonClass(object):\n            _classMachine = MethodicalMachine()\n        '
    moduleDict = self.makeModuleAsDict(source, self.pathDir, 'clsmod.py')
    PythonClass = moduleDict['clsmod.PythonClass']
    self.assertIn(('clsmod.PythonClass._classMachine', PythonClass.load()._classMachine), list(self.findMachinesViaWrapper(PythonClass)))