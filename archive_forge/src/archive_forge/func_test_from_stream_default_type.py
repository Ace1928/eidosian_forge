import io
import os
import tempfile
import unittest
from testtools import TestCase
from testtools.compat import (
from testtools.content import (
from testtools.content_type import (
from testtools.matchers import (
from testtools.tests.helpers import an_exc_info
def test_from_stream_default_type(self):
    data = io.StringIO('some data')
    content = content_from_stream(data)
    self.assertThat(content.content_type, Equals(UTF8_TEXT))