import re
from breezy import (branch, controldir, directory_service, errors, osutils,
from breezy.bzr import bzrdir, knitrepo
from breezy.tests import http_server, scenarios, script, test_foreign
from breezy.transport import memory
def test_push_dash_d(self):
    t = self.make_branch_and_tree('from')
    t.commit(allow_pointless=True, message='first commit')
    self.run_bzr('push -d from to-one')
    self.assertPathExists('to-one')
    self.run_bzr('push -d %s %s' % tuple(map(urlutils.local_path_to_url, ['from', 'to-two'])))
    self.assertPathExists('to-two')