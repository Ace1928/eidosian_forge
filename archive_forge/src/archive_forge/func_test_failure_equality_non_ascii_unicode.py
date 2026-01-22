import pickle
import sys
from oslo_utils import encodeutils
from taskflow import exceptions
from taskflow import test
from taskflow.tests import utils as test_utils
from taskflow.types import failure
def test_failure_equality_non_ascii_unicode(self):
    hi_ru = u'привет'
    fail = failure.Failure.from_exception(ValueError(hi_ru))
    copied = fail.copy()
    self.assertEqual(fail, copied)