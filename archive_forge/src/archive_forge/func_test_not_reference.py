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
def test_not_reference(self):
    opener = self.make_branch_opener(False, ['a', None])
    self.assertEqual('a', opener.check_and_follow_branch_reference('a'))
    self.assertEqual(['a'], opener.follow_reference_calls)