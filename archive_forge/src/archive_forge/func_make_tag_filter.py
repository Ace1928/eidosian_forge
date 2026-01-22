import csv
import datetime
import testtools
from testtools import StreamResult
from testtools.content import TracebackContent, text_content
import iso8601
import subunit
def make_tag_filter(with_tags, without_tags):
    """Make a callback that checks tests against tags."""
    with_tags = with_tags and set(with_tags) or None
    without_tags = without_tags and set(without_tags) or None

    def check_tags(test, outcome, err, details, tags):
        if with_tags and (not with_tags <= tags):
            return False
        if without_tags and bool(without_tags & tags):
            return False
        return True
    return check_tags