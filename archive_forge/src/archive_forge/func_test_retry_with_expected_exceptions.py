import time
from unittest import mock
import eventlet
from eventlet.green import threading as greenthreading
from oslotest import base as test_base
from oslo_service import fixture
from oslo_service import loopingcall
def test_retry_with_expected_exceptions(self):
    result = 'RESULT'
    responses = [AnException(None), AnException(None), result]

    def func(*args, **kwargs):
        response = responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response
    sleep_time_incr = 0.01
    retry_count = 2
    retry = loopingcall.RetryDecorator(10, sleep_time_incr, 10, (AnException,))
    self.assertEqual(result, retry(func)())
    self.assertTrue(retry._retry_count == retry_count)
    self.assertEqual(retry_count * sleep_time_incr, retry._sleep_time)