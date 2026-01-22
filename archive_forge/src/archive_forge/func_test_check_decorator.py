from unittest import mock
import oslotest.base as base
from osc_placement import version
def test_check_decorator(self):
    fake_api = mock.Mock()
    fake_api_dec = version.check(version.gt('2.11'))(fake_api)
    obj = mock.Mock()
    obj.app.client_manager.placement.api_version = '2.12'
    fake_api_dec(obj, 1, 2, 3)
    fake_api.assert_called_once_with(obj, 1, 2, 3)
    fake_api.reset_mock()
    obj.app.client_manager.placement.api_version = '2.10'
    self.assertRaisesRegex(ValueError, 'Operation or argument is not supported', fake_api_dec, obj, 1, 2, 3)
    fake_api.assert_not_called()