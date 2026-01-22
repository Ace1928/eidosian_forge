import io
import logging
import time
import testtools
from testtools import TestCase
from fixtures import (
def test_replace_and_restore_handlers(self):
    stream = io.StringIO()
    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler(stream))
    logger.setLevel(logging.INFO)
    logging.info('one')
    fixture = LogHandler(self.CustomHandler())
    with fixture:
        logging.info('two')
    logging.info('three')
    self.assertEqual(['two'], fixture.handler.msgs)
    self.assertEqual('one\nthree\n', stream.getvalue())