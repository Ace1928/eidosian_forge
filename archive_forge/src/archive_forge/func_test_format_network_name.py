import re
from io import BytesIO
from ... import branch as _mod_branch
from ... import commit, controldir
from ... import delta as _mod_delta
from ... import errors, gpg, info, repository
from ... import revision as _mod_revision
from ... import tests, transport, upgrade, workingtree
from ...bzr import branch as _mod_bzrbranch
from ...bzr import inventory, knitpack_repo, remote
from ...bzr import repository as bzrrepository
from .. import per_repository, test_server
from ..matchers import *
def test_format_network_name(self):
    repo = self.make_repository('r')
    format = repo._format
    network_name = format.network_name()
    self.assertIsInstance(network_name, bytes)
    if isinstance(format, remote.RemoteRepositoryFormat):
        repo._ensure_real()
        real_repo = repo._real_repository
        self.assertEqual(real_repo._format.network_name(), network_name)
    else:
        registry = repository.network_format_registry
        looked_up_format = registry.get(network_name)
        self.assertEqual(format.__class__, looked_up_format.__class__)