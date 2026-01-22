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
def test___init___sets_content_type(self):
    stack_lines, expected = self._get_stack_line_and_expected_output()
    content = StackLinesContent(stack_lines)
    expected_content_type = ContentType('text', 'x-traceback', {'language': 'python', 'charset': 'utf8'})
    self.assertEqual(expected_content_type, content.content_type)