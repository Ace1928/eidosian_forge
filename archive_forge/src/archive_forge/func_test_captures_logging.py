import io
import logging
import time
import testtools
from testtools import TestCase
from fixtures import (
def test_captures_logging(self):
    fixture = self.useFixture(LogHandler(self.CustomHandler()))
    logging.info('some message')
    self.assertEqual(['some message'], fixture.handler.msgs)