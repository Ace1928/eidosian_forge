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
def get_chrooted_scheme(self, relpath):
    """Create a server that is chrooted to `relpath`.

        :return: ``(scheme, get_url)`` where ``scheme`` is the scheme of the
            chroot server and ``get_url`` returns URLs on said server.
        """
    transport = self.get_transport(relpath)
    chroot_server = chroot.ChrootServer(transport)
    chroot_server.start_server()
    self.addCleanup(chroot_server.stop_server)

    def get_url(relpath):
        return chroot_server.get_url() + relpath
    return (urlutils.URL.from_string(chroot_server.get_url()).scheme, get_url)