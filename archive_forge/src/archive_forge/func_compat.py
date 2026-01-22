import csv
import datetime
import testtools
from testtools import StreamResult
from testtools.content import TracebackContent, text_content
import iso8601
import subunit
def compat(test, outcome, error, details, tags):
    try:
        return filter_predicate(test, outcome, error, details, tags)
    except TypeError:
        return filter_predicate(test, outcome, error, details)