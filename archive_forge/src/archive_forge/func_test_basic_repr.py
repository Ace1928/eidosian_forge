from testtools import TestCase
from testtools.matchers import Equals, MatchesException, Raises
from testtools.content_type import (
def test_basic_repr(self):
    content_type = ContentType('text', 'plain')
    self.assertThat(repr(content_type), Equals('text/plain'))