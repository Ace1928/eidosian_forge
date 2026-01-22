import re
from breezy import (branch, controldir, directory_service, errors, osutils,
from breezy.bzr import bzrdir, knitrepo
from breezy.tests import http_server, scenarios, script, test_foreign
from breezy.transport import memory
def make_dummy_builder(self, relpath):
    builder = self.make_branch_builder(relpath, format=test_foreign.DummyForeignVcsDirFormat())
    builder.build_snapshot(None, [('add', ('', b'TREE_ROOT', 'directory', None)), ('add', ('foo', b'fooid', 'file', b'bar'))], revision_id=b'revid')
    return builder