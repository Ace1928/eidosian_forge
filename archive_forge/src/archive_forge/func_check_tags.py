import csv
import datetime
import testtools
from testtools import StreamResult
from testtools.content import TracebackContent, text_content
import iso8601
import subunit
def check_tags(test, outcome, err, details, tags):
    if with_tags and (not with_tags <= tags):
        return False
    if without_tags and bool(without_tags & tags):
        return False
    return True