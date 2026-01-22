import os
from testtools import content
from .. import plugins as _mod_plugins
from .. import trace
from ..bzr.smart import medium
from ..controldir import ControlDir
from ..transport import remote
from . import TestCaseWithTransport
def test_import_tariffs_working(self):
    self.make_branch_and_tree('.')
    self.run_command_check_imports(['st'], ['nonexistentmodulename', 'anothernonexistentmodule'])
    self.assertRaises(AssertionError, self.run_command_check_imports, ['st'], ['breezy.tree'])