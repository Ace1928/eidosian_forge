from doctest import ELLIPSIS
from pprint import pformat
import sys
import _thread
import unittest
from testtools import (
from testtools.compat import (
from testtools.content import (
from testtools.matchers import (
from testtools.testcase import (
from testtools.testresult.doubles import (
from testtools.tests.helpers import (
from testtools.tests.samplecases import (
def test_unique_text_generator(self):
    prefix = self.getUniqueString()
    unique_text_generator = testcase.unique_text_generator(prefix)
    first_result = next(unique_text_generator)
    self.assertEqual('{}-{}'.format(prefix, 'Ḁ'), first_result)
    second_result = next(unique_text_generator)
    self.assertEqual('{}-{}'.format(prefix, 'ḁ'), second_result)