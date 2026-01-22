import os
from io import BytesIO
from .. import bedding, ignores
from . import TestCase, TestCaseInTempDir, TestCaseWithTransport
def test_add_duplicate(self):
    """Adding the same ignore twice shouldn't add a new entry."""
    ignores.add_runtime_ignores(['foo', 'bar'])
    self.assertEqual({'foo', 'bar'}, ignores.get_runtime_ignores())
    ignores.add_runtime_ignores(['bar'])
    self.assertEqual({'foo', 'bar'}, ignores.get_runtime_ignores())