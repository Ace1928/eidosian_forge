import re
from breezy import (branch, controldir, directory_service, errors, osutils,
from breezy.bzr import bzrdir, knitrepo
from breezy.tests import http_server, scenarios, script, test_foreign
from breezy.transport import memory
def test_push_from_subdir(self):
    t = self.make_branch_and_tree('tree')
    self.build_tree(['tree/dir/', 'tree/dir/file'])
    t.add(['dir', 'dir/file'])
    t.commit('r1')
    out, err = self.run_bzr('push ../../pushloc', working_dir='tree/dir')
    self.assertEqual('', out)
    self.assertEqual('Created new branch.\n', err)