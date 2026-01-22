import re
from breezy import (branch, controldir, directory_service, errors, osutils,
from breezy.bzr import bzrdir, knitrepo
from breezy.tests import http_server, scenarios, script, test_foreign
from breezy.transport import memory
def test_push_redirects_on_mkdir(self):
    """If the push requires a mkdir, push respects redirect requests.

        This is added primarily to handle lp:/ URI support, so that users can
        push to new branches by specifying lp:/ URIs.
        """
    destination_url = self.memory_server.get_url() + 'source'
    self.run_bzr(['push', '-d', 'tree', destination_url])
    local_revision = branch.Branch.open('tree').last_revision()
    remote_revision = branch.Branch.open(self.memory_server.get_url() + 'target').last_revision()
    self.assertEqual(remote_revision, local_revision)