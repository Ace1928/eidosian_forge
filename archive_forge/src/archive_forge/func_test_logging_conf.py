import datetime
import logging
import logging.config
import os
import sys
from oslo_utils import timeutils
import testscenarios
import oslo_messaging
from oslo_messaging.tests import utils as test_utils
from unittest import mock
@mock.patch('oslo_utils.timeutils.utcnow')
def test_logging_conf(self, mock_utcnow):
    fake_transport = oslo_messaging.get_notification_transport(self.conf)
    with mock.patch('oslo_messaging.transport._get_transport', return_value=fake_transport):
        logging.config.dictConfig({'version': 1, 'handlers': {'notification': {'class': 'oslo_messaging.LoggingNotificationHandler', 'level': self.priority.upper(), 'url': 'test://'}}, 'loggers': {'default': {'handlers': ['notification'], 'level': self.priority.upper()}}})
    mock_utcnow.return_value = datetime.datetime.utcnow()
    levelno = getattr(logging, self.priority.upper())
    logger = logging.getLogger('default')
    lineno = sys._getframe().f_lineno + 1
    logger.log(levelno, 'foobar')
    n = oslo_messaging.notify._impl_test.NOTIFICATIONS[0][1]
    self.assertEqual(getattr(self, 'queue', self.priority.upper()), n['priority'])
    self.assertEqual('logrecord', n['event_type'])
    self.assertEqual(str(timeutils.utcnow()), n['timestamp'])
    self.assertIsNone(n['publisher_id'])
    pathname = __file__
    if pathname.endswith(('.pyc', '.pyo')):
        pathname = pathname[:-1]
    self.assertDictEqual(n['payload'], {'process': os.getpid(), 'funcName': 'test_logging_conf', 'name': 'default', 'thread': None, 'levelno': levelno, 'processName': 'MainProcess', 'pathname': pathname, 'lineno': lineno, 'msg': 'foobar', 'exc_info': None, 'levelname': logging.getLevelName(levelno), 'extra': None})