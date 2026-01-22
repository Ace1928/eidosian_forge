import contextlib
from breezy import branch as _mod_branch
from breezy import config, controldir
from breezy import delta as _mod_delta
from breezy import (errors, lock, merge, osutils, repository, revision, shelf,
from breezy import tree as _mod_tree
from breezy import urlutils
from breezy.bzr import remote
from breezy.tests import per_branch
from breezy.tests.http_server import HttpServer
from breezy.transport import memory
def test_create_checkout(self):
    tree_a = self.make_branch_and_tree('a')
    branch_a = tree_a.branch
    checkout_b = branch_a.create_checkout('b')
    self.assertEqual(b'null:', checkout_b.last_revision())
    try:
        rev1 = checkout_b.commit('rev1')
    except errors.NoRoundtrippingSupport:
        raise tests.TestNotApplicable('roundtripping between %r and %r not supported' % (checkout_b.branch, checkout_b.branch.get_master_branch()))
    self.assertEqual(rev1, branch_a.last_revision())
    self.assertNotEqual(checkout_b.branch.base, branch_a.base)
    checkout_c = branch_a.create_checkout('c', lightweight=True)
    self.assertEqual(rev1, checkout_c.last_revision())
    rev2 = checkout_c.commit('rev2')
    self.assertEqual(rev2, branch_a.last_revision())
    self.assertEqual(checkout_c.branch.base, branch_a.base)
    checkout_d = branch_a.create_checkout('d', lightweight=True)
    self.assertEqual(rev2, checkout_d.last_revision())
    checkout_e = branch_a.create_checkout('e')
    self.assertEqual(rev2, checkout_e.last_revision())