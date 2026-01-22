from testtools import TestCase
from testtools.matchers import Equals, MatchesException, Raises
from testtools.content_type import (
def test_extended_repr(self):
    content_type = ContentType('text', 'plain', {'foo': 'bar', 'baz': 'qux'})
    self.assertThat(repr(content_type), Equals('text/plain; baz="qux"; foo="bar"'))