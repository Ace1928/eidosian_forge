from unittest import mock
from oslotest import base as test_base
from oslo_service import fixture
from oslo_service import loopingcall
def test_sleep_fixture(self):

    @loopingcall.RetryDecorator(max_retry_count=3, inc_sleep_time=2, exceptions=(ValueError,))
    def retried_method():
        raise ValueError('!')
    self.assertRaises(ValueError, retried_method)
    self.assertEqual(3, self.sleepfx.mock_wait.call_count)
    self.sleepfx.mock_wait.assert_has_calls([mock.call(x) for x in (2, 4, 6)])