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
def test_iter_text_default_charset_iso_8859_1(self):
    content_type = ContentType('text', 'strange')
    text = 'bytesê'
    iso_version = text.encode('ISO-8859-1')
    content = Content(content_type, lambda: [iso_version])
    self.assertEqual([text], list(content.iter_text()))