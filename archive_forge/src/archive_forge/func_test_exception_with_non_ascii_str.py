import pickle
import sys
from oslo_utils import encodeutils
from taskflow import exceptions
from taskflow import test
from taskflow.tests import utils as test_utils
from taskflow.types import failure
def test_exception_with_non_ascii_str(self):
    bad_string = chr(200)
    excp = ValueError(bad_string)
    fail = failure.Failure.from_exception(excp)
    self.assertEqual(encodeutils.exception_to_unicode(excp), fail.exception_str)
    expected = u'Failure: ValueError: È'
    self.assertEqual(expected, str(fail))