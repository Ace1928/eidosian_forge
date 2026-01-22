import testtools
import fixtures
from fixtures import (
def test_adds_missing_to_end_package_path(self):
    uniquedir = self.useFixture(TempDir()).path
    fixture = PackagePathEntry('fixtures', uniquedir)
    self.assertFalse(uniquedir in fixtures.__path__)
    with fixture:
        self.assertTrue(uniquedir in fixtures.__path__)
    self.assertFalse(uniquedir in fixtures.__path__)