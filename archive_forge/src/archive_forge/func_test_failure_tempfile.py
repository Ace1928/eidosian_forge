import os
import sys
import tempfile
from .. import mergetools, tests
def test_failure_tempfile(self):

    def dummy_invoker(exe, args, cleanup):
        self._exe = exe
        self._args = args
        self.assertPathExists(args[0])
        self.log(repr(args))
        with open(args[0], 'w') as f:
            self.log(repr(f))
            f.write('temp stuff')
        cleanup(1)
        return 1
    retcode = mergetools.invoke('tool {this_temp}', 'test.txt', dummy_invoker)
    self.assertEqual(1, retcode)
    self.assertEqual('tool', self._exe)
    self.assertFileEqual(b'stuff', 'test.txt')