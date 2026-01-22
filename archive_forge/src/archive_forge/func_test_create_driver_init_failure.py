from unittest import mock
from osprofiler import notifier
from osprofiler.tests import test
@mock.patch('osprofiler.notifier.base.get_driver')
def test_create_driver_init_failure(self, mock_get_driver):
    mock_get_driver.side_effect = Exception()
    result = notifier.create('test', 10, b=20)
    mock_get_driver.assert_called_once_with('test', 10, b=20)
    self.assertEqual(notifier._noop_notifier, result)