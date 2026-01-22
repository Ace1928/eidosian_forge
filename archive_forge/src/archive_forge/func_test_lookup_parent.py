from .. import transport, urlutils
from ..directory_service import (AliasDirectory, DirectoryServiceRegistry,
from . import TestCase, TestCaseWithTransport
def test_lookup_parent(self):
    self.assertAliasFromBranch(self.branch.set_parent, 'http://a', ':parent')