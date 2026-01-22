import csv
import datetime
import testtools
from testtools import StreamResult
from testtools.content import TracebackContent, text_content
import iso8601
import subunit
def wasSuccessful(self):
    """Tells whether or not this result was a success"""
    return self.failed_tests == 0