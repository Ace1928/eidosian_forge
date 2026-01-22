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
def test_postfix_content(self):
    stack_lines, expected = self._get_stack_line_and_expected_output()
    postfix = '\n' + self.getUniqueString()
    content = StackLinesContent(stack_lines, postfix_content=postfix)
    actual = content.as_text()
    expected = expected + postfix
    self.assertEqual(expected, actual)