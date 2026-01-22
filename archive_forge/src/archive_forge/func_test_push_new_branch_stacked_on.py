import re
from breezy import (branch, controldir, directory_service, errors, osutils,
from breezy.bzr import bzrdir, knitrepo
from breezy.tests import http_server, scenarios, script, test_foreign
from breezy.transport import memory
def test_push_new_branch_stacked_on(self):
    """Pushing a new branch with --stacked-on creates a stacked branch."""
    trunk_tree, branch_tree = self.create_trunk_and_feature_branch()
    out, err = self.run_bzr(['push', '--stacked-on', trunk_tree.branch.base, self.get_url('published')], working_dir='branch')
    self.assertEqual('', out)
    self.assertEqual('Created new stacked branch referring to %s.\n' % trunk_tree.branch.base, err)
    self.assertPublished(branch_tree.last_revision(), trunk_tree.branch.base)