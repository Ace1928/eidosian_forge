import re
from breezy import (branch, controldir, directory_service, errors, osutils,
from breezy.bzr import bzrdir, knitrepo
from breezy.tests import http_server, scenarios, script, test_foreign
from breezy.transport import memory
def test_push_verbose_shows_log(self):
    tree = self.make_branch_and_tree('source')
    tree.commit('rev1')
    out, err = self.run_bzr('push -v -d source target')
    self.assertContainsRe(out, 'rev1')
    tree.commit('rev2')
    out, err = self.run_bzr('push -v -d source target')
    self.assertContainsRe(out, 'rev2')
    self.assertNotContainsRe(out, 'rev1')