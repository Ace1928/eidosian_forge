import re
from breezy import (branch, controldir, directory_service, errors, osutils,
from breezy.bzr import bzrdir, knitrepo
from breezy.tests import http_server, scenarios, script, test_foreign
from breezy.transport import memory
def make_local_branch_and_tree(self):
    self.tree = self.make_branch_and_tree('local')
    self.build_tree_contents([('local/file', b'initial')])
    self.tree.add('file')
    self.tree.commit('adding file', rev_id=b'added')
    self.build_tree_contents([('local/file', b'modified')])
    self.tree.commit('modify file', rev_id=b'modified')