import os
import shutil
import stat
import sys
from ...controldir import ControlDir
from .. import KnownFailure, TestCaseWithTransport, TestSkipped
def testWorkflow(self):
    """Run through a basic usage scenario."""
    self.run_bzr(['bisect', 'start'])
    self.run_bzr(['bisect', 'yes'])
    self.run_bzr(['bisect', 'no', '-r', '1'])
    self.assertRevno(3)
    self.run_bzr(['bisect', 'yes'])
    self.assertRevno(2)
    self.run_bzr(['bisect', 'no'])
    self.assertRevno(3)
    self.run_bzr(['bisect', 'no'])
    self.assertRevno(3)