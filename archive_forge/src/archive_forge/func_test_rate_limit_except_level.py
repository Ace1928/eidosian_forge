import io
import logging
from unittest import mock
from oslotest import base as test_base
from oslo_log import rate_limit
@mock.patch('oslo_log.rate_limit.monotonic_clock')
def test_rate_limit_except_level(self, mock_clock):
    mock_clock.return_value = 1
    logger, stream = self.install_filter(1, 1, 'CRITICAL')
    logger.error('error 1')
    logger.error('error 2')
    logger.critical('critical 3')
    logger.critical('critical 4')
    self.assertEqual(stream.getvalue(), 'error 1\nLogging rate limit: drop after 1 records/1 sec\ncritical 3\ncritical 4\n')