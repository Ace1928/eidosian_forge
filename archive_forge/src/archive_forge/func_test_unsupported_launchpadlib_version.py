from ... import bedding, errors, osutils
from ...tests import TestCase, TestCaseWithTransport
from ...tests.features import ModuleAvailableFeature
def test_unsupported_launchpadlib_version(self):
    launchpadlib = launchpadlib_feature.module
    self.patch(launchpadlib, '__version__', '1.5.0')
    self.lp_api.MINIMUM_LAUNCHPADLIB_VERSION = (1, 5, 1)
    self.assertRaises(errors.DependencyNotPresent, self.lp_api.check_launchpadlib_compatibility)