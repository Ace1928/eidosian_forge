import io
import logging
import time
import testtools
from testtools import TestCase
from fixtures import (
def test_logging_output_included_in_details(self):
    fixture = FakeLogger()
    detail_name = "pythonlogging:''"
    with fixture:
        content = fixture.getDetails()[detail_name]
        logging.info('some message')
        self.assertEqual('some message\n', content.as_text())
    self.assertEqual('some message\n', content.as_text())
    with fixture:
        self.assertEqual('', fixture.getDetails()[detail_name].as_text())
    try:
        self.assertEqual('some message\n', content.as_text())
    except AssertionError:
        raise
    except:
        pass