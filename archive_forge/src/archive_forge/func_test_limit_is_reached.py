import builtins
import functools
import io
import time
from unittest import mock
import ddt
from os_brick import exception
from os_brick.tests import base
from os_brick import utils
def test_limit_is_reached(self):
    self.counter = 0
    retries = 3
    interval = 2
    backoff_rate = 4
    with mock.patch.object(utils, '_time_sleep') as mock_sleep:

        @utils.retry(exception.VolumeDeviceNotFound, interval, retries, backoff_rate)
        def always_fails():
            self.counter += 1
            raise exception.VolumeDeviceNotFound(device='fake')
        self.assertRaises(exception.VolumeDeviceNotFound, always_fails)
        self.assertEqual(retries, self.counter)
        expected_sleep_arg = []
        for i in range(retries):
            if i > 0:
                interval *= backoff_rate ** (i - 1)
                expected_sleep_arg.append(float(interval))
        mock_sleep.assert_has_calls(list(map(mock.call, expected_sleep_arg)))