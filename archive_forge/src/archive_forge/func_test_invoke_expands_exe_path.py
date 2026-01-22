import os
import sys
import tempfile
from .. import mergetools, tests
def test_invoke_expands_exe_path(self):
    self.overrideEnv('PATH', os.path.dirname(sys.executable))

    def dummy_invoker(exe, args, cleanup):
        self._exe = exe
        self._args = args
        cleanup(0)
        return 0
    command = '%s {result}' % os.path.basename(sys.executable)
    retcode = mergetools.invoke(command, 'test.txt', dummy_invoker)
    self.assertEqual(0, retcode)
    self.assertEqual(sys.executable, self._exe)
    self.assertEqual(['test.txt'], self._args)