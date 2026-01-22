from .. import transport, urlutils
from ..directory_service import (AliasDirectory, DirectoryServiceRegistry,
from . import TestCase, TestCaseWithTransport
def test_lookup_non_default(self):
    default = self.make_branch('.')
    non_default = default.controldir.create_branch(name='nondefault')
    self.assertEqual(non_default.base, directories.dereference('co:nondefault'))