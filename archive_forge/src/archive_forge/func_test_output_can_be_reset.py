import io
import logging
import time
import testtools
from testtools import TestCase
from fixtures import (
def test_output_can_be_reset(self):
    fixture = FakeLogger()
    with fixture:
        logging.info('message')
    fixture.reset_output()
    self.assertEqual('', fixture.output)