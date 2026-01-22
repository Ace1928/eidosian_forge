from .. import transport, urlutils
from ..directory_service import (AliasDirectory, DirectoryServiceRegistry,
from . import TestCase, TestCaseWithTransport
def test_lookup_bound(self):
    self.assertAliasFromBranch(self.branch.set_bound_location, 'http://d', ':bound')