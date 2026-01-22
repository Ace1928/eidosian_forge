from unittest import mock
import oslotest.base as base
from osc_placement import version
def test_check_mixin(self):

    class Test(version.CheckerMixin):
        app = mock.Mock()
        app.client_manager.placement.api_version = '1.2'
    t = Test()
    self.assertTrue(t.compare_version(version.le('1.3')))
    self.assertTrue(t.check_version(version.ge('1.0')))
    self.assertRaisesRegex(ValueError, 'Operation or argument is not supported', t.check_version, version.lt('1.2'))