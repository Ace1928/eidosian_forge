import codecs
import io
import sys
import traceback
from unittest import skipIf
from twisted.python.compat import (
from twisted.python.filepath import FilePath
from twisted.python.runtime import platform
from twisted.trial.unittest import SynchronousTestCase, TestCase
def test_execfileGlobalsAndLocals(self):
    """
        L{execfile} executes the specified file in the given global and local
        namespaces.
        """
    script = self.writeScript('foo += 1\n')
    globalNamespace = {'foo': 10}
    localNamespace = {'foo': 20}
    execfile(script.path, globalNamespace, localNamespace)
    self.assertEqual(10, globalNamespace['foo'])
    self.assertEqual(21, localNamespace['foo'])