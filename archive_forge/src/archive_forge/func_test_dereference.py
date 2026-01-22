from .. import transport, urlutils
from ..directory_service import (AliasDirectory, DirectoryServiceRegistry,
from . import TestCase, TestCaseWithTransport
def test_dereference(self):
    self.assertEqual(OldService.base + 'bar', self.registry.dereference('old:bar'))
    self.assertEqual(OldService.base + 'bar', self.registry.dereference('old:bar', purpose='write'))
    self.assertEqual('baz:qux', self.registry.dereference('baz:qux'))
    self.assertEqual('baz:qux', self.registry.dereference('baz:qux', purpose='write'))