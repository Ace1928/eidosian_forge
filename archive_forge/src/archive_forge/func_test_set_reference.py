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
def test_set_reference(self):
    """set_reference on all regular branches should be callable."""
    if not self.branch_format.is_supported():
        return
    this_branch = self.make_branch('this')
    other_branch = self.make_branch('other')
    try:
        this_branch._format.set_reference(this_branch.controldir, None, other_branch)
    except (NotImplementedError, errors.IncompatibleFormat):
        pass
    else:
        ref = this_branch._format.get_reference(this_branch.controldir)
        self.assertEqual(ref, other_branch.user_url)