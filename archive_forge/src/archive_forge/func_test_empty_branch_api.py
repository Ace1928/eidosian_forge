import os
from io import BytesIO
from ... import (branch, builtins, check, controldir, errors, push, revision,
from ...bzr import branch as bzrbranch
from ...bzr.smart import client
from .. import per_branch, test_server
def test_empty_branch_api(self):
    """The branch_obj.push API should make a limited number of HPSS calls.
        """
    t = transport.get_transport_from_url(self.smart_server.get_url()).clone('target')
    target = branch.Branch.open_from_transport(t)
    self.empty_branch.push(target)
    self.assertEqual([b'BzrDir.open_2.1', b'BzrDir.open_branchV3', b'BzrDir.find_repositoryV3', b'Branch.get_stacked_on_url', b'Branch.lock_write', b'Branch.last_revision_info', b'Branch.unlock'], self.hpss_calls)