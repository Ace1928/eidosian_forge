import re
from breezy import (branch, controldir, directory_service, errors, osutils,
from breezy.bzr import bzrdir, knitrepo
from breezy.tests import http_server, scenarios, script, test_foreign
from breezy.transport import memory
def test_push_without_tree(self):
    b = self.make_branch('.')
    out, err = self.run_bzr('push pushed-location')
    self.assertEqual('', out)
    self.assertEqual('Created new branch.\n', err)
    b2 = branch.Branch.open('pushed-location')
    self.assertEndsWith(b2.base, 'pushed-location/')