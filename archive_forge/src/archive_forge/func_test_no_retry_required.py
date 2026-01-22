import builtins
import functools
import io
import time
from unittest import mock
import ddt
from os_brick import exception
from os_brick.tests import base
from os_brick import utils
def test_no_retry_required(self):
    self.counter = 0
    with mock.patch.object(utils, '_time_sleep') as mock_sleep:

        @utils.retry(exception.VolumeDeviceNotFound, interval=2, retries=3, backoff_rate=2)
        def succeeds():
            self.counter += 1
            return 'success'
        ret = succeeds()
        self.assertFalse(mock_sleep.called)
        self.assertEqual('success', ret)
        self.assertEqual(1, self.counter)