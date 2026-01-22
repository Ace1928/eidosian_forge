from unittest import mock
import uuid
import testtools
from openstack import connection
from openstack import exceptions
from openstack.tests.unit import base
from openstack import utils
@mock.patch('time.sleep')
def test_iterate_timeout_str_wait(self, mock_sleep):
    iter = utils.iterate_timeout(10, 'test_iterate_timeout_str_wait', wait='1.6')
    next(iter)
    next(iter)
    mock_sleep.assert_called_with(1.6)