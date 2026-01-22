import logging
import warnings
import pytest
import cirq.testing
def test_assert_logs_log_level():
    with cirq.testing.assert_logs('apple'):
        logging.error('orange apple fruit')
        logging.debug('should not')
        logging.info('count')
    with cirq.testing.assert_logs('apple', 'critical', count=2):
        logging.critical('critical')
        logging.error('orange apple fruit')
        logging.debug('should not')
        logging.info('count')
    with cirq.testing.assert_logs('apple', min_level=logging.INFO, count=2):
        logging.error('orange apple fruit')
        logging.debug('should not')
        logging.info('count')
    with cirq.testing.assert_logs('info only 1', min_level=logging.INFO, max_level=logging.INFO):
        with cirq.testing.assert_logs('info warning 1', min_level=logging.WARNING, max_level=logging.WARNING):
            logging.info('info only 1')
            logging.warning('info warning 1')