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
def setup_redirect(self, target_url):

    class RedirectingTransport(Transport):

        def get(self, name):
            raise RedirectRequested(self.base, target_url)

        def _redirected_to(self, source, target):
            return get_transport(target)
    register_transport_proto('redirecting://', help='Test transport that redirects.')
    register_transport('redirecting://', RedirectingTransport)
    self.addCleanup(unregister_transport, 'redirecting://', RedirectingTransport)