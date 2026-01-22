import io
import logging
import time
import testtools
from testtools import TestCase
from fixtures import (
def test_logging_level_restored(self):
    self.logger.setLevel(logging.DEBUG)
    fixture = LogHandler(self.CustomHandler(), level=logging.WARNING)
    with fixture:
        logging.debug('debug message')
        self.assertEqual(logging.WARNING, self.logger.level)
    self.assertEqual([], fixture.handler.msgs)
    self.assertEqual(logging.DEBUG, self.logger.level)