from ... import bedding, errors, osutils
from ...tests import TestCase, TestCaseWithTransport
from ...tests.features import ModuleAvailableFeature
def test_get_launchpadlib_version(self):
    version_info = self.lp_api.parse_launchpadlib_version('1.5.1')
    self.assertEqual((1, 5, 1), version_info)