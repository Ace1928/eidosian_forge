import re
from breezy import (branch, controldir, directory_service, errors, osutils,
from breezy.bzr import bzrdir, knitrepo
from breezy.tests import http_server, scenarios, script, test_foreign
from breezy.transport import memory
def test_push_no_tree(self):
    b = self.make_branch_and_tree('push-from')
    self.build_tree(['push-from/file'])
    b.add('file')
    b.commit('commit 1')
    out, err = self.run_bzr('push --no-tree -d push-from push-to')
    self.assertEqual('', out)
    self.assertEqual('Created new branch.\n', err)
    self.assertPathDoesNotExist('push-to/file')