import csv
import datetime
import testtools
from testtools import StreamResult
from testtools.content import TracebackContent, text_content
import iso8601
import subunit
def reportTest(self, test_id, duration):
    if self.show_times:
        seconds = duration.seconds
        seconds += duration.days * 3600 * 24
        seconds += duration.microseconds / 1000000.0
        self._stream.write(test_id + ' %0.3f\n' % seconds)
    else:
        self._stream.write(test_id + '\n')