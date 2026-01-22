import csv
import datetime
import testtools
from testtools import StreamResult
from testtools.content import TracebackContent, text_content
import iso8601
import subunit
Construct a ``TestByTestResult``.

        :param on_test: A callable that take a test case, a status (one of
            "success", "failure", "error", "skip", or "xfail"), a start time
            (a ``datetime`` with timezone), a stop time, an iterable of tags,
            and a details dict. Is called at the end of each test (i.e. on
            ``stopTest``) with the accumulated values for that test.
        