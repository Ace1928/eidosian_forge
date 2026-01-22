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
def test_create_colocated(self):
    try:
        repo = self.make_repository('.', shared=True)
    except errors.IncompatibleFormat:
        return
    if repo.controldir._format.colocated_branches:
        raise tests.TestNotApplicable('control dir does not support colocated branches')
    self.assertEqual(0, len(repo.controldir.list_branches()))
    if not self.bzrdir_format.colocated_branches:
        raise tests.TestNotApplicable('control dir format does not support colocated branches')
    try:
        child_branch1 = self.branch_format.initialize(repo.controldir, name='branch1')
    except errors.UninitializableFormat:
        return
    self.assertEqual(1, len(repo.controldir.list_branches()))
    self.branch_format.initialize(repo.controldir, name='branch2')
    self.assertEqual(2, len(repo.controldir.list_branches()))