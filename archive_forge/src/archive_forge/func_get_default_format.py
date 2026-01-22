from breezy.bzr.bzrdir import BzrDirFormat
from breezy.controldir import ControlDirFormat
from breezy.tests import (TestCaseWithTransport, default_transport,
from breezy.tests.per_controldir import make_scenarios
from breezy.transport import memory
def get_default_format(self):
    return self.bzrdir_format