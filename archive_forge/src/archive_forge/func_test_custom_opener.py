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
def test_custom_opener(self):
    a = self.make_branch('a', format='2a')
    b = self.make_branch('b', format='2a')
    b.set_stacked_on_url(a.base)
    TrackingProber.seen_urls = []
    opener = self.make_branch_opener([a.base, b.base], probers=[TrackingProber])
    opener.open(b.base)
    self.assertEqual(set(TrackingProber.seen_urls), {b.base, a.base})