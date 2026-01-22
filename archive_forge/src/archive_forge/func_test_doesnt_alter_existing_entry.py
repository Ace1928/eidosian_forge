import testtools
import fixtures
from fixtures import (
def test_doesnt_alter_existing_entry(self):
    existingdir = fixtures.__path__[0]
    expectedlen = len(fixtures.__path__)
    fixture = PackagePathEntry('fixtures', existingdir)
    with fixture:
        self.assertTrue(existingdir in fixtures.__path__)
        self.assertEqual(expectedlen, len(fixtures.__path__))
    self.assertTrue(existingdir in fixtures.__path__)
    self.assertEqual(expectedlen, len(fixtures.__path__))