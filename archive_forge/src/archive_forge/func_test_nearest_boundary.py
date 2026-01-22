from unittest import mock
from testtools import matchers
from oslo_service import periodic_task
from oslo_service.tests import base
@mock.patch('oslo_service.periodic_task.now')
@mock.patch('random.random')
def test_nearest_boundary(self, mock_random, mock_now):
    mock_now.return_value = 19
    mock_random.return_value = 0
    self.assertEqual(17, periodic_task._nearest_boundary(10, 7))
    mock_now.return_value = 28
    self.assertEqual(27, periodic_task._nearest_boundary(13, 7))
    mock_now.return_value = 1841
    self.assertEqual(1837, periodic_task._nearest_boundary(781, 88))
    mock_now.return_value = 1835
    self.assertEqual(mock_now.return_value, periodic_task._nearest_boundary(None, 88))
    mock_random.return_value = 1.0
    mock_now.return_value = 1300
    self.assertEqual(1200 + 10, periodic_task._nearest_boundary(1000, 200))
    mock_random.return_value = 0.5
    mock_now.return_value = 1300
    self.assertEqual(1200 + 5, periodic_task._nearest_boundary(1000, 200))