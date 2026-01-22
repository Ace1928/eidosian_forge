import io
import sys
from unittest import mock
from urllib import parse
from novaclient import base
from novaclient import exceptions
from novaclient.tests.unit import fakes
from novaclient.tests.unit import utils as test_utils
from novaclient import utils
from novaclient.v2 import servers
def test_record_time(self):
    times = []
    with utils.record_time(times, True, 'a', 'b'):
        pass
    self.assertEqual(1, len(times))
    self.assertEqual(3, len(times[0]))
    self.assertEqual('a b', times[0][0])
    self.assertIsInstance(times[0][1], float)
    self.assertIsInstance(times[0][2], float)
    times = []
    with utils.record_time(times, False, 'x'):
        pass
    self.assertEqual(0, len(times))