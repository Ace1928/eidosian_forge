from typing import List
from .. import urlutils
from ..branch import Branch
from ..bzr import BzrProber
from ..bzr.branch import BranchReferenceFormat
from ..controldir import ControlDir, ControlDirFormat
from ..errors import NotBranchError, RedirectRequested
from ..transport import (Transport, chroot, get_transport, register_transport,
from ..url_policy_open import (BadUrl, BranchLoopError, BranchOpener,
from . import TestCase, TestCaseWithTransport
def test_hook_does_not_interfere(self):
    self.make_branch('stacked')
    self.make_branch('stacked-on')
    Branch.open('stacked').set_stacked_on_url('../stacked-on')
    Branch.open('stacked')