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
def test_self_stacked_branch(self):
    a = self.make_branch('a', format='1.6')
    a.get_config().set_user_option('stacked_on_location', a.base)
    opener = self.make_branch_opener([a.base])
    self.assertRaises(BranchLoopError, opener.open, a.base)