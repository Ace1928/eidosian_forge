from os import chdir, devnull, getcwd
from subprocess import PIPE, Popen
from sys import executable
from twisted.python.filepath import FilePath
from twisted.python.modules import getModule
from twisted.python.test.test_shellcomp import ZshScriptTestMixin
from twisted.trial.unittest import SkipTest, TestCase
def test_twistdPathInsert(self):
    """
        The twistd script adds the current working directory to sys.path so
        that it's able to import modules from it.
        """
    script = self.bin.child('twistd')
    if not script.exists():
        raise SkipTest('Script tests do not apply to installed configuration.')
    cwd = getcwd()
    self.addCleanup(chdir, cwd)
    testDir = FilePath(self.mktemp())
    testDir.makedirs()
    chdir(testDir.path)
    testDir.child('bar.tac').setContent('import sys\nprint sys.path\n')
    output = outputFromPythonScript(script, '-ny', 'bar.tac')
    self.assertIn(repr(testDir.path), output)