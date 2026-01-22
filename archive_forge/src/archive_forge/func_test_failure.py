import os
import sys
import tempfile
from .. import mergetools, tests
def test_failure(self):

    def dummy_invoker(exe, args, cleanup):
        self._exe = exe
        self._args = args
        cleanup(1)
        return 1
    retcode = mergetools.invoke('tool {result}', 'test.txt', dummy_invoker)
    self.assertEqual(1, retcode)
    self.assertEqual('tool', self._exe)
    self.assertEqual(['test.txt'], self._args)