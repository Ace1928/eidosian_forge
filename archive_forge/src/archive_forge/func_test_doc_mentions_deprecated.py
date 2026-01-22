from testtools import TestCase
from testtools.matchers import Contains
from fixtures import (
def test_doc_mentions_deprecated(self):
    self.assertThat(DetailStream.__doc__, Contains('Deprecated'))