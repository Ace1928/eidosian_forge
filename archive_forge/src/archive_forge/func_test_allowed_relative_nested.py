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
def test_allowed_relative_nested(self):
    self.get_transport().mkdir('subdir')
    a = self.make_branch('subdir/a', format='1.6')
    b = self.make_branch('b', format='1.6')
    b.set_stacked_on_url('../subdir/a')
    c = self.make_branch('subdir/c', format='1.6')
    c.set_stacked_on_url('../../b')
    opener = self.make_branch_opener([c.base, b.base, a.base])
    opener.open(c.base)