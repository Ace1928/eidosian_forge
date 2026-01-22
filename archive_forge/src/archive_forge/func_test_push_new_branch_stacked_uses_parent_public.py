import re
from breezy import (branch, controldir, directory_service, errors, osutils,
from breezy.bzr import bzrdir, knitrepo
from breezy.tests import http_server, scenarios, script, test_foreign
from breezy.transport import memory
def test_push_new_branch_stacked_uses_parent_public(self):
    """Pushing a new branch with --stacked creates a stacked branch."""
    trunk_tree, branch_tree = self.create_trunk_and_feature_branch()
    self.transport_readonly_server = http_server.HttpServer
    trunk_public = self.make_branch('public_trunk', format='1.9')
    trunk_public.pull(trunk_tree.branch)
    trunk_public_url = self.get_readonly_url('public_trunk')
    br = trunk_tree.branch
    br.set_public_branch(trunk_public_url)
    out, err = self.run_bzr(['push', '--stacked', self.get_url('published')], working_dir='branch')
    self.assertEqual('', out)
    self.assertEqual('Created new stacked branch referring to %s.\n' % trunk_public_url, err)
    self.assertPublished(branch_tree.last_revision(), trunk_public_url)