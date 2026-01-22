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
def test_get_reference(self):
    """get_reference on all regular branches should return None."""
    if not self.branch_format.is_supported():
        return
    made_controldir = self.make_controldir('.')
    made_controldir.create_repository()
    if made_controldir._format.colocated_branches:
        name = 'foo'
    else:
        name = None
    try:
        made_branch = made_controldir.create_branch(name)
    except errors.UninitializableFormat:
        raise tests.TestNotApplicable('Uninitializable branch format')
    self.assertEqual(None, made_branch._format.get_reference(made_branch.controldir, name))