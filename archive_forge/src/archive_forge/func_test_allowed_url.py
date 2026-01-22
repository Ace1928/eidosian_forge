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
def test_allowed_url(self):
    stacked_on_branch = self.make_branch('base-branch', format='1.6')
    stacked_branch = self.make_branch('stacked-branch', format='1.6')
    stacked_branch.set_stacked_on_url(stacked_on_branch.base)
    opener = self.make_branch_opener([stacked_branch.base, stacked_on_branch.base])
    opener.open(stacked_branch.base)