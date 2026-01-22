import operator
import os
import shutil
import sys
import textwrap
import tempfile
from unittest import skipIf, TestCase
def test_discoverAll(self):
    """
        Given a top-level package FQPN, L{findMachines} discovers all
        L{MethodicalMachine} instances in and below it.
        """
    machines = sorted(self.findMachines('test_package'), key=operator.itemgetter(0))
    tpRootLevel = self.packageDict['test_package.rootLevel'].load()
    tpPythonClass = self.packageDict['test_package.PythonClass'].load()
    mRLAttr = self.moduleDict['test_package.subpackage.module.rootLevel']
    mRootLevel = mRLAttr.load()
    mPCAttr = self.moduleDict['test_package.subpackage.module.PythonClass']
    mPythonClass = mPCAttr.load()
    expectedMachines = sorted([('test_package.rootLevel', tpRootLevel), ('test_package.PythonClass._machine', tpPythonClass._machine), ('test_package.subpackage.module.rootLevel', mRootLevel), ('test_package.subpackage.module.PythonClass._machine', mPythonClass._machine)], key=operator.itemgetter(0))
    self.assertEqual(expectedMachines, machines)